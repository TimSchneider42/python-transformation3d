"""
MIT License

Copyright (c) 2022 Tim Schneider

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Union, Sequence, Optional, List, Dict

import numpy as np
from scipy.spatial.transform import Rotation


class Transformation:
    def __init__(self, translation: Optional[Sequence[Union[Sequence[float], float]]] = None,
                 rotation: Optional[Rotation] = None):
        translation = np.zeros((3,)) if translation is None else np.asarray(translation)
        rotation = Rotation.from_quat(np.array([0, 0, 0, 1])) if rotation is None else rotation
        if len(translation.shape) not in [1, 2] or translation.shape[-1] != 3:
            raise ValueError("Translation must be an array of shape Nx3 or 3.")
        translation_single = len(translation.shape) == 1
        rotation_single = rotation.single
        if rotation_single and not translation_single:
            rotation = Rotation.from_quat(rotation.as_quat()[None])
        elif translation_single and not rotation_single:
            translation = translation[None]
        if not rotation_single or not translation_single:
            td = translation.shape[0]
            rd = len(rotation)
            if td == 1 and rd != 1:
                translation = np.repeat(translation.reshape((-1, 3)), rd, axis=0)
            elif rd == 1 and td != 1:
                rotation = Rotation.from_quat(np.repeat(rotation.as_quat().reshape((-1, 4)), td, axis=0))
            if td != rd and td != 1 and rd != 1:
                raise ValueError(
                    "Translation and rotation have incompatible batch dimensions ({} vs {}).".format(td, rd))
        self.__translation = translation
        self.__rotation = rotation

    def transform(self, other: Union["Transformation", np.ndarray], inverse: bool = False) \
            -> Union["Transformation", np.ndarray]:
        if isinstance(other, np.ndarray):
            return self._transform_positions(other, inverse=inverse)

        output_pos = self._transform_positions(np.array(other.translation), inverse=inverse)
        output_rot = self.__rotation * other.rotation if not inverse else self.__rotation.inv() * other.rotation

        return Transformation(output_pos, output_rot)

    def copy(self) -> "Transformation":
        return Transformation.from_matrix(self.matrix)

    def to_dict(self):
        return {
            "translation": self.__translation.tolist(),
            "rotation": self.__rotation.as_quat().tolist()
        }

    def _transform_positions(self, positions: np.ndarray, inverse: bool = False):
        if not inverse:
            return self.__rotation.apply(positions) + self.__translation
        else:
            return self.__rotation.apply((positions - self.__translation), inverse=True)

    @classmethod
    def from_pos_quat(cls, position: Optional[Sequence[float]] = None,
                      quaternion: Optional[Sequence[float]] = None) -> "Transformation":
        return cls(position, None if quaternion is None else Rotation.from_quat(quaternion))

    @classmethod
    def from_pos_euler(cls, position: Optional[Sequence[float]] = None,
                       euler_angles: Optional[Sequence[float]] = None, sequence: str = "xyz") -> "Transformation":
        return cls(position, None if euler_angles is None else Rotation.from_euler(sequence, euler_angles))

    @classmethod
    def from_pos_rotvec(cls, position: Optional[Sequence[float]] = None,
                        rotvec: Optional[Sequence[float]] = None) -> "Transformation":
        return cls(position, None if rotvec is None else Rotation.from_rotvec(rotvec))

    @classmethod
    def from_matrix(cls, matrix: Union[List, np.ndarray]) -> "Transformation":
        matrix = np.asarray(matrix)
        translation = matrix[..., :3, 3]
        rotation = Rotation.from_matrix(matrix[..., :3, :3])
        return cls(translation, rotation)

    @classmethod
    def from_dict(cls, transformation_dict: Dict[str, Sequence[float]]) -> "Transformation":
        return Transformation.from_pos_quat(transformation_dict["translation"], transformation_dict["rotation"])

    @property
    def rotation(self) -> Rotation:
        return self.__rotation

    @property
    def translation(self) -> np.ndarray:
        return self.__translation

    @property
    def quaternion(self) -> np.ndarray:
        return self.__rotation.as_quat()

    @property
    def angle(self) -> float:
        q = self.__rotation.as_quat()
        return np.abs(2 * np.arctan2(np.linalg.norm(q[..., :-1], axis=-1), np.abs(q[..., -1])))

    @property
    def rotvec(self) -> np.ndarray:
        return self.__rotation.as_rotvec()

    @property
    def matrix(self) -> np.ndarray:
        if self.single:
            output = np.zeros((4, 4))
        else:
            output = np.zeros((len(self), 4, 4))
        output[..., :3, :3] = self.__rotation.as_matrix()
        output[..., 3, 3] = 1
        output[..., :3, 3] = self.__translation
        return output

    @property
    def inv(self) -> "Transformation":
        pos = - self.__rotation.apply(self.__translation, inverse=True)
        rot = self.__rotation.inv()
        return Transformation(pos, rot)

    @property
    def single(self):
        return self.__rotation.single

    def __mul__(self, other: Union["Transformation", Sequence["Transformation"], np.ndarray]):
        return self.transform(other)

    def __matmul__(self, other: Union["Transformation", Sequence["Transformation"], np.ndarray]):
        return self.transform(other)

    def __getitem__(self, item: Union[int, slice]):
        return Transformation(self.__translation[item], self.__rotation[item])

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __len__(self):
        return None if self.single else self.__translation.shape[0]

    def __repr__(self):
        return "Transformation({}, {})".format(self.__translation.tolist(), self.__rotation.as_quat().tolist())
