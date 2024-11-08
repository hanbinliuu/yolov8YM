#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Runs protoc with the gRPC plugin to generate messages and gRPC stubs."""

import os

import pkg_resources
from grpc_tools import protoc


def gen_proto_modules(filename, path):
    """执行grpc的转化"""
    proto_include = pkg_resources.resource_filename('grpc_tools', '_proto')
    proto_file = f'{path}/{filename}'
    print(f'protoc process {proto_file} with include path={proto_include}, {path}')
    protoc.main((
        '',
        '-I{}'.format(path),  # include path
        '-I{}'.format(proto_include),  # include google/protobuf/*.proto
        '--python_out=./protos',  # *_pb2.py file
        '--grpc_python_out=./protos',  # Stubs: *_pb2_grpc.py file
        proto_file,
    ))


def file_name(file_dir):
    """获取当前文件夹中所有的proto文件"""
    _protos = []
    for f in os.listdir(file_dir):
        if f[-5:] == 'proto':
            _protos.append(f)
    return _protos


if __name__ == '__main__':
    list_protofile = file_name(os.path.join(os.getcwd(), "protos"))
    for protofile in list_protofile:
        gen_proto_modules(protofile, './protos')
