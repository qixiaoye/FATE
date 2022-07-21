#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


import fate_tensor.par
import torch

from ._metaclass import (
    PHEBlockMetaclass,
    phe_decryptor_metaclass,
    phe_encryptor_metaclass,
    phe_keygen_metaclass,
)


class PaillierBlock(metaclass=PHEBlockMetaclass):
    pass


class BlockPaillierEncryptor(
    metaclass=phe_encryptor_metaclass(PaillierBlock, torch.Tensor)
):
    pass


class BlockPaillierDecryptor(
    metaclass=phe_decryptor_metaclass(PaillierBlock, torch.Tensor)
):
    pass


class BlockPaillierCipher(
    metaclass=phe_keygen_metaclass(
        BlockPaillierEncryptor, BlockPaillierDecryptor, fate_tensor.par.keygen
    )
):
    pass
