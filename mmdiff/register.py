import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torch.fft import fft2, ifft2
def adain(cnt_feat, sty_feat):
    cnt_mean = cnt_feat.mean(dim=[0, 2],keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 2],keepdim=True)
    sty_mean = sty_feat.mean(dim=[0, 2],keepdim=True)
    sty_std = sty_feat.std(dim=[0, 2],keepdim=True)
    output = ((cnt_feat-cnt_mean)/cnt_std)*sty_std + sty_mean
    return output


class ResidualStore:
    def __init__(self, save_path):
        self.num_res_layers = 0
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.current_timestep = None
        self.injected_residuals = {}  # 用于存储注入的残差特征
        self.save_mode = True  # 是否存储残差特征
        self.replace_mode = False  # 是否替换残差特征
        self.layer_range = None

    def set_save_mode(self, mode: bool):
        """设置是否存储残差特征"""
        self.save_mode = mode

    def set_replace_mode(self, mode: bool):
        """设置是否替换残差特征"""
        self.replace_mode = mode

    def set_timestep(self, timestep):
        """设置当前时间步"""
        self.current_timestep = timestep

    def store_residual(self, residual, place_in_unet, layer_idx):
        """保存残差特征到文件"""
        if not self.save_mode:  # 如果当前是非保存模式，直接返回
            return
        if self.current_timestep is None:
            raise ValueError("Timestep is not set. Call `set_timestep` before storing residuals.")
        # 创建保存路径
        timestep_dir = os.path.join(self.save_path, f"timestep_{self.current_timestep}")
        if not os.path.exists(timestep_dir):
            os.makedirs(timestep_dir)
        file_path = os.path.join(timestep_dir, f"{place_in_unet}_resnet_{layer_idx}.pt")
        torch.save(residual.cpu(), file_path)

    def load_residual(self, timestep, place_in_unet, layer_idx, device=None):
        """加载残差特征并确保其在正确的设备上"""
        file_path = os.path.join(self.save_path, f"timestep_{timestep}", f"{place_in_unet}_resnet_{layer_idx}.pt")
        # print("Load:", file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Residual file not found: {file_path}")
        residual = torch.load(file_path)
        if device is not None:
            residual = residual.to(device)  # 将残差特征移动到指定设备
        return residual

    def inject_residual(self, timestep, place_in_unet, layer_idx, residual):
        """注入指定时间步和层的残差特征"""
        if timestep not in self.injected_residuals:
            self.injected_residuals[timestep] = {}
        if place_in_unet not in self.injected_residuals[timestep]:
            self.injected_residuals[timestep][place_in_unet] = {}
        self.injected_residuals[timestep][place_in_unet][layer_idx] = residual

    def inject_residuals_for_range(self, timesteps, place_in_unet, layer_range, device):
        """注入多个时间步和层范围的残差特征"""
        self.layer_range = layer_range
        for timestep in timesteps: # 981
            for layer_idx in self.layer_range:  # [0,1,2,3,4]
                try:
                    residual = self.load_residual(timestep, place_in_unet, layer_idx, device=device)
                    self.inject_residual(timestep, place_in_unet, layer_idx, residual)
                except FileNotFoundError as e:
                    print(f"File not found for timestep {timestep}, layer {layer_idx}: {e}")
                except Exception as e:
                    print(f"Error loading residual for timestep {timestep}, layer {layer_idx}: {e}")

    def get_inject_residual(self, timestep, place_in_unet, layer_idx):
        # 检查时间步是否存在
        if timestep not in self.injected_residuals:
            # print(f"[DEBUG] No residuals injected for timestep {timestep}")
            return None

        # 检查 UNet 位置是否存在
        if place_in_unet not in self.injected_residuals[timestep]:
            # print(f"[DEBUG] No residuals for place_in_unet '{place_in_unet}' at timestep {timestep}")
            return None

        # 检查层索引是否存在
        if layer_idx not in self.injected_residuals[timestep][place_in_unet]:
            # print(f"[DEBUG] No residuals for layer {layer_idx} at timestep {timestep}, place_in_unet '{place_in_unet}'")
            return None

        # 获取残差特征
        residual = self.injected_residuals[timestep][place_in_unet][layer_idx]

        if residual is None:
            # print(f"[DEBUG] Residual is None for timestep {timestep}, layer {layer_idx}")
            return None

        return residual

    # def get_inject_residual(self, timestep, place_in_unet, layer_idx):
    #     """获取注入的残差特征"""
    #     if (
    #         timestep in self.injected_residuals
    #         and place_in_unet in self.injected_residuals[timestep]
    #         and layer_idx in self.injected_residuals[timestep][place_in_unet]
    #     ):
    #         return self.injected_residuals[timestep][place_in_unet][layer_idx]
    #     return None

def register_resnet_control(model, controller):
    def ca_forward(self, place_in_unet, layer_idx=None):
        # 保存原始的 forward 方法
        original_forward = self.forward

        def forward(hidden_states, temb=None):
            # 调用原始的 forward 方法，获取输出
            output = original_forward(hidden_states, temb)

            if place_in_unet == 'up':
                # 保存残差特征到文件（仅在 save_mode 为 True 时）
                if controller.save_mode:
                    controller.store_residual(output, place_in_unet, layer_idx)

            if place_in_unet == 'up' and layer_idx is not None and controller.layer_range is not None:
                if layer_idx in controller.layer_range:
                    # 检查是否需要替换残差特征
                    if controller.replace_mode:
                        injected_residual = controller.get_inject_residual(
                            controller.current_timestep, place_in_unet, layer_idx)
                        if injected_residual is not None:
                            # # 使用 AdaIN 对 injected_residual 和 output 进行风格对齐

                            output = injected_residual

            return output

        return forward


    # 如果未提供 controller，则报错
    if controller is None:
        raise ValueError("No controller, you must set a controller!")

    def register_recr(net_, count, place_in_unet, layer_idx=0):
        if net_.__class__.__name__ == 'ResnetBlock2D':
            net_.forward = ca_forward(net_, place_in_unet, layer_idx)
            layer_idx += 1
            return count+1, layer_idx
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count , layer_idx = register_recr(net__, count, place_in_unet, layer_idx)
        return count, layer_idx

    resnet_down_count = 0
    resnet_mid_count = 0
    resnet_up_count = 0
    sub_nets = model.named_children()

    for net in sub_nets:
        # net[0]代表该层的姓名，net[1]代表对应的层
        # 例如：('time_proj', Timesteps())
        if "down" in net[0]:  # 8
            layer_down_idx = 0
            resnet_down_count, layer_down_idx = register_recr(net[1], 0, "down", layer_down_idx)
        elif "up" in net[0]:  # 12
            layer_up_idx = 0
            resnet_up_count, layer_up_idx = register_recr(net[1], 0, "up", layer_up_idx)
        elif "mid" in net[0]:  # 2
            layer_mid_idx = 0
            resnet_mid_count, layer_mid_idx = register_recr(net[1], 0, "mid", layer_mid_idx)

    controller.num_res_layers = resnet_down_count + resnet_up_count + resnet_mid_count

class AttentionStore:
    def __init__(self, save_path):
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.num_att_layers = 0
        self.cross_att_count = 0
        self.self_att_count = 0
        self.current_timestep = None
        self.save_mode = True  # 是否保存特征
        self.replace_mode = False  # 是否替换特征
        self.file_counters = {"self_attention": 0, "cross_attention": 0}  # 分开计数器
        self.attn_layers = []  # 存储需要替换的层索引
        self.components = []  # 存储需要替换的组件
        self.timesteps_to_inject = []

    def set_save_mode(self, mode: bool):
        """设置是否保存注意力特征"""
        self.save_mode = mode

    def set_replace_mode(self, mode: bool):
        """设置是否替换注意力特征"""
        self.replace_mode = mode

    def set_timestep(self, timestep):
        """设置当前时间步"""
        self.current_timestep = timestep
        self.file_counters = {"self_attention": 0, "cross_attention": 0}  # 每次设置新时间步时重置计数器

    def set_timesteps_to_inject(self, timesteps):
        """设置需要注入特征的时间步范围"""
        self.timesteps_to_inject = timesteps

    def should_inject(self):
        """检查当前时间步是否在注入范围内"""
        return self.current_timestep in self.timesteps_to_inject

    def set_replace_layers(self, attn_layers, components):
        """
        设置需要替换的层和组件。
        attn_layers: list of layer indices to replace, e.g., [1]
        components: list of components to replace, e.g., ['q', 'k']
        """
        self.attn_layers = attn_layers
        self.components = components

    def _get_save_directory(self, attention_type, component, layer_index):
        """
        根据时间步、注意力类型、层索引和特征组件生成保存路径。
        """
        if self.current_timestep is None:
            raise ValueError("Timestep is not set. Call set_timestep before saving features.")
        timestep_dir = os.path.join(self.save_path, f"timestep_{self.current_timestep}")
        component_dir = os.path.join(timestep_dir, attention_type, component)
        if not os.path.exists(component_dir):
            os.makedirs(component_dir)
        return component_dir

    def save_features(self, query, attention_type, place_in_unet, layer_index):
        """
        保存注意力特征到文件。
        """
        if not self.save_mode:
            return

        # 确定保存目录
        q_dir = self._get_save_directory(attention_type, "q", layer_index)

        # 更新计数器（按注意力类型独立管理）
        self.file_counters[attention_type] += 1
        counter = self.file_counters[attention_type]

        # 保存特征
        torch.save(query.cpu(), os.path.join(q_dir, f"up_{counter}.pt"))

    def load_features(self, attention_type, time, layer_index, device=None):
        """
        加载指定层的注意力特征。
        """
        features = {}
        for component in self.components:
            # 构建文件路径
            feature_path = os.path.join(
                self.save_path,
                f"timestep_{time}",
                attention_type,
                component,
                f"up_{layer_index}.pt"
            )
            try:
                # 加载特征
                feature = torch.load(feature_path)
                if device is not None:
                    feature = feature.to(device)
                # print(f"Loaded {component} feature from layer {layer_index}: {feature.shape}")

                # 存储到字典
                features[component] = feature

            except FileNotFoundError:
                print(f"File not found: {feature_path}. Skipping {component}.")
            except Exception as e:
                print(f"Error loading {component} feature: {e}")

        return features

def register_attention_control(model, controller):
    def attention_forward(self, layer_index, place_in_unet, attention_type):
        # 不需要保存 to_out 了
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            residual = hidden_states

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, kwargs.get('temb', None))

            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            # 如果是“up”阶段，并且需要替换特征
            if place_in_unet == 'up' and controller.should_inject() and controller.replace_mode and layer_index in controller.attn_layers:

                # 加载需要替换的特征
                features = controller.load_features(attention_type, time=controller.current_timestep,
                                                    layer_index=layer_index, device=hidden_states.device)
                for component in controller.components:
                    if component == 'q':
                        query = features['q']
                        # query = adain(query_,query)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # 直接处理输出部分
            hidden_states = self.to_out[0](hidden_states)
            hidden_states = self.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            # 保存特征
            if place_in_unet == 'up' and controller.save_mode:
                controller.save_features(
                    query=query,
                    attention_type=attention_type,
                    place_in_unet=place_in_unet,
                    layer_index=layer_index
                )

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor
            return hidden_states

        return forward

    if controller is None:
        raise ValueError("No controller, you must set a controller!")

    def register_recr(net_, count, place_in_unet, layer_idx):
        if net_.__class__.__name__ == 'Attention':
            if net_.to_q.in_features == net_.to_k.in_features:  # Self-attention
                net_.forward = attention_forward(net_, layer_idx, place_in_unet, "self_attention")
                controller.self_att_count += 1
                return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet, layer_idx=count + 1)
        return count

    att_count = 0
    sub_nets = model.named_children()
    for net in sub_nets:
        if "up" in net[0]:
            att_count += register_recr(net[1], 0, "up", layer_idx=att_count + 1)

    controller.num_att_layers = att_count
    # print(f"Total Attention Layers: {att_count}")
    # print(f"Cross-Attention Layers: {controller.cross_att_count}")
    # print(f"Self-Attention Layers: {controller.self_att_count}")

def register_single_channel_decoder(vae_instance):
    """
    替换 VAE 的 _decode 方法，使其解码为单通道图像。

    Args:
        vae_instance: 目标 VAE 实例，其 _decode 方法需要被替换。
    """
    import types
    import torch
    from typing import Union
    from diffusers.models.autoencoders.vae import DecoderOutput

    def new_decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            return self.tiled_decode(z, return_dict=return_dict)

        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)

        dec = self.decoder(z)
        # 转换为单通道
        dec = dec.mean(dim=1, keepdim=True)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    # 动态替换实例的 _decode 方法
    vae_instance._decode = types.MethodType(new_decode, vae_instance)