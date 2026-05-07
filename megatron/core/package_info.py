# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.


MAJOR = 0
MINOR = 16
PATCH = 0
PRE_RELEASE = 'rc0'

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = '.'.join(map(str, VERSION[:3]))
__version__ = '.'.join(map(str, VERSION[:3])) + ''.join(VERSION[3:])

__package_name__ = 'unipool_megatron'
__contact_names__ = 'UniPool contributors'
__contact_emails__ = ''
__homepage__ = 'https://github.com/UniPool/UniPool'
__repository_url__ = 'https://github.com/UniPool/UniPool'
__download_url__ = 'https://github.com/UniPool/UniPool/releases'
__description__ = (
    'UniPool research fork of Megatron Core with shared expert-pool MoE training'
)
__license__ = 'BSD-3'
__keywords__ = (
    'deep learning, machine learning, gpu, NLP, MoE, UniPool, transformer, pytorch, torch'
)
