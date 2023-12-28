import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512

LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
        prompt: str,
        uncod_prompt: str, # 생성하지 않았으면 하는 조건
        input_image = None,
        strength = 0.8,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name="ddpm",
        n_inference_steps=50,
        models={}, # 사용할 모델들을 사전형으로 저장
        seed=None,
        device=None,
        idle_device=None,
        tokenizer=None,
    ):
    with torch.no_grad():
        if not 0<strength <=1:
            raise ValueError("strength must be between 0 and 1")

        # cuda 뭐 이런거 지정하는 함수를 정의하는거 일 듯
        '''idle device가 쉬고있는 device니까,
        특정 작업이 끝나면 GPU에 올라온 모델을 CPU나 다른 쉬고있는 Device로 옮겨주는 함수임!'''
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)

        # random seed 설정
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (batch_size, seq_len)
            # condition prompt
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # clip에 텍스트 입력
            # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
            cond_context = clip(cond_tokens)

            # uncondition prompt
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncod_prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)

            # 그냥 concat 해벼려..?
            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)
        # clip을 GPU로..?
        to_idle(clip)

        if sampler_name == "ddpm":
            '''sampler 객체 생성'''
            '''같은 난수 발생기를 입력으로 넣어줌 (generator)'''
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        latent_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            # resize image to 512, 512
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            # (Height, Width, Channel)
            input_image_tensor = np.array(input_image_tensor)
            # np.array to tensor
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            input_image_tensor = input_image_tensor.unsqueeze(0) # 차원 삽입: for batch
            # 모델의 입력으로 들어갈 수 있도록
            # (batch_size, Height, Width, Channel) -> (batch_size, Channel, Height, Width) 로 변경
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # random sampler torch.generator를 사용해서 random_noise 생성
            encoder_noise = torch.randn(latent_shape, generator=generator, device=device)
            '''입력 이미지와 랜덤 노이지를 같이 encoder에 넣음'''
            '''(batch_size, 4, latents_height, latents_width)'''
            latents = encoder(input_image_tensor, encoder_noise)

            '''
            latent에 노이즈 추가!
            sampler는 DDPM임
            '''

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)

        else:
            # 이미지 없으면 걍 생 랜덤 latent noise 생성
            latents = torch.randn(latent_shape, generator=generator, device=device)

        diffusion = models['diffusion']
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps) # ddpm's timestep

        for i, timestep in enumerate(timesteps):
            # (1, 320)
            # timestep을 입력하면 time_embedding을 가져오는 함수
            time_embedding = get_time_embedding(timestep).to(device)

            model_input = latents

            if do_cfg: # do_cfg: negative prompt가 있다는 뜻.
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1) # batch를 2배로...?

            model_output = diffusion(model_input, context, time_embedding) # denoising 한 결과 sampling

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2) # 두개로 나눔 (conditional output, uncond output)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # latent 갱신?
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        # latent를 pixel space의 이미지로 변환
        images = decoder(latents)
        to_idle(decoder)

        # 다시 reverse pre-process
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x-=old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # 총 160개의 timestep 이 있는건강...?
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # Shape: (1, 160)
    # 텐서 index에 None을 넣으면 unsqueeze와 같은 효과 ㅎㄷㄷ
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)