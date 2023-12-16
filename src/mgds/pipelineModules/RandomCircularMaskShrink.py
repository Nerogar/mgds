import torch
from torch import Tensor

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class RandomCircularMaskShrink(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            mask_name: str,
            enabled_in_name: str,
            shrink_probability: float,
            shrink_factor_min: float,
            shrink_factor_max: float = 1
    ):
        super(RandomCircularMaskShrink, self).__init__()

        self.mask_name = mask_name
        self.enabled_in_name = enabled_in_name
        self.shrink_probability = shrink_probability
        self.shrink_factor_min = shrink_factor_min
        self.shrink_factor_max = shrink_factor_max

    def length(self) -> int:
        return self._get_previous_length(self.mask_name)

    def get_inputs(self) -> list[str]:
        return [self.mask_name]

    def get_outputs(self) -> list[str]:
        return [self.mask_name]

    @staticmethod
    def __get_random_point_in_mask(mask: Tensor, seed: int) -> (int, int):
        # Generate a random point within a binary tensor by
        # 1. generating a new tensor with random uniform numbers in the same shape as the mask
        # 2. multiplying the mask with the random tensor
        # 3. using argmax, get the point with the highest random number within the masked region

        generator = torch.Generator(device=mask.device)
        generator.manual_seed(seed)
        random = torch.rand(size=mask.shape, generator=generator, dtype=torch.float32, device=mask.device)
        random_mask = torch.flatten(random * mask)
        max_index_flat = torch.argmax(random_mask).item()
        max_index = (max_index_flat // random.shape[2], max_index_flat % random.shape[2])

        return max_index

    @staticmethod
    def __get_radial_gradient(mask: Tensor, center: (int, int)) -> Tensor:
        # Generate a radial gradient where each pixel is calculated as sqrt(x^2 + y^2) by
        # 1. calculating and squaring a gradient in each direction
        # 2. adding both gradients together
        # 3. taking the square root of the resulting gradient

        resolution = (mask.shape[1], mask.shape[2])
        top = -center[0]
        bottom = resolution[0] - center[0] - 1
        left = -center[1]
        right = resolution[1] - center[1] - 1

        vertical_gradient = torch.linspace(start=top, end=bottom, steps=resolution[0], dtype=torch.float32,
                                           device=mask.device)
        vertical_gradient = vertical_gradient * vertical_gradient
        vertical_gradient = vertical_gradient.unsqueeze(1)
        vertical_gradient = vertical_gradient.expand(resolution)

        horizontal_gradient = torch.linspace(start=left, end=right, steps=resolution[1], dtype=torch.float32,
                                             device=mask.device)
        horizontal_gradient = horizontal_gradient * horizontal_gradient
        horizontal_gradient = horizontal_gradient.unsqueeze(0)
        horizontal_gradient = horizontal_gradient.expand(resolution)

        radial_gradient = torch.sqrt(vertical_gradient + horizontal_gradient)
        radial_gradient = radial_gradient.unsqueeze(0)

        radial_gradient = radial_gradient.to(dtype=mask.dtype)

        return radial_gradient

    @staticmethod
    def __get_disc_mask(radial_gradient: Tensor, radius) -> Tensor:
        return (radial_gradient <= radius).to(dtype=radial_gradient.dtype)

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        enabled = self._get_previous_item(variation, self.enabled_in_name, index)
        mask = self._get_previous_item(variation, self.mask_name, index)

        if enabled:
            rand = self._get_rand(variation, index)

            random_center = self.__get_random_point_in_mask(mask, rand.randint(0, 1 << 30))
            radial_gradient = self.__get_radial_gradient(mask, random_center)

            max_radius = (mask * radial_gradient).max().item()
            radius = rand.uniform(self.shrink_factor_min, self.shrink_factor_max) * max_radius

            disc_mask = self.__get_disc_mask(radial_gradient, radius)

            result_mask = mask * disc_mask

            return {
                self.mask_name: result_mask,
            }
        else:
            return {
                self.mask_name: mask,
            }
