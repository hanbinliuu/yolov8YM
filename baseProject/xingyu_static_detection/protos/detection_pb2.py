# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: detection.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='detection.proto',
  package='com.yimuzn.featuresdetection.device.grpc',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0f\x64\x65tection.proto\x12(com.yimuzn.featuresdetection.device.grpc\"\"\n\rDetectRequest\x12\x11\n\timagePath\x18\x01 \x03(\t\"3\n\x0c\x46\x65\x61tureCount\x12\x14\n\x0c\x66\x65\x61ture_name\x18\x01 \x01(\x05\x12\r\n\x05\x63ount\x18\x02 \x01(\x05\"\xab\x01\n\x0e\x44\x65tectResponse\x12\x63\n\x0e\x66\x65\x61ture_counts\x18\x01 \x03(\x0b\x32K.com.yimuzn.featuresdetection.device.grpc.DetectResponse.FeatureCountsEntry\x1a\x34\n\x12\x46\x65\x61tureCountsEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x32\x99\x01\n\rDetectService\x12\x87\x01\n\x12\x44\x65tectFeaturesYolo\x12\x37.com.yimuzn.featuresdetection.device.grpc.DetectRequest\x1a\x38.com.yimuzn.featuresdetection.device.grpc.DetectResponseb\x06proto3'
)




_DETECTREQUEST = _descriptor.Descriptor(
  name='DetectRequest',
  full_name='com.yimuzn.featuresdetection.device.grpc.DetectRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='imagePath', full_name='com.yimuzn.featuresdetection.device.grpc.DetectRequest.imagePath', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=61,
  serialized_end=95,
)


_FEATURECOUNT = _descriptor.Descriptor(
  name='FeatureCount',
  full_name='com.yimuzn.featuresdetection.device.grpc.FeatureCount',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='feature_name', full_name='com.yimuzn.featuresdetection.device.grpc.FeatureCount.feature_name', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='count', full_name='com.yimuzn.featuresdetection.device.grpc.FeatureCount.count', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=97,
  serialized_end=148,
)


_DETECTRESPONSE_FEATURECOUNTSENTRY = _descriptor.Descriptor(
  name='FeatureCountsEntry',
  full_name='com.yimuzn.featuresdetection.device.grpc.DetectResponse.FeatureCountsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='com.yimuzn.featuresdetection.device.grpc.DetectResponse.FeatureCountsEntry.key', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='com.yimuzn.featuresdetection.device.grpc.DetectResponse.FeatureCountsEntry.value', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=270,
  serialized_end=322,
)

_DETECTRESPONSE = _descriptor.Descriptor(
  name='DetectResponse',
  full_name='com.yimuzn.featuresdetection.device.grpc.DetectResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='feature_counts', full_name='com.yimuzn.featuresdetection.device.grpc.DetectResponse.feature_counts', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_DETECTRESPONSE_FEATURECOUNTSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=151,
  serialized_end=322,
)

_DETECTRESPONSE_FEATURECOUNTSENTRY.containing_type = _DETECTRESPONSE
_DETECTRESPONSE.fields_by_name['feature_counts'].message_type = _DETECTRESPONSE_FEATURECOUNTSENTRY
DESCRIPTOR.message_types_by_name['DetectRequest'] = _DETECTREQUEST
DESCRIPTOR.message_types_by_name['FeatureCount'] = _FEATURECOUNT
DESCRIPTOR.message_types_by_name['DetectResponse'] = _DETECTRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

DetectRequest = _reflection.GeneratedProtocolMessageType('DetectRequest', (_message.Message,), {
  'DESCRIPTOR' : _DETECTREQUEST,
  '__module__' : 'detection_pb2'
  # @@protoc_insertion_point(class_scope:com.yimuzn.featuresdetection.device.grpc.DetectRequest)
  })
_sym_db.RegisterMessage(DetectRequest)

FeatureCount = _reflection.GeneratedProtocolMessageType('FeatureCount', (_message.Message,), {
  'DESCRIPTOR' : _FEATURECOUNT,
  '__module__' : 'detection_pb2'
  # @@protoc_insertion_point(class_scope:com.yimuzn.featuresdetection.device.grpc.FeatureCount)
  })
_sym_db.RegisterMessage(FeatureCount)

DetectResponse = _reflection.GeneratedProtocolMessageType('DetectResponse', (_message.Message,), {

  'FeatureCountsEntry' : _reflection.GeneratedProtocolMessageType('FeatureCountsEntry', (_message.Message,), {
    'DESCRIPTOR' : _DETECTRESPONSE_FEATURECOUNTSENTRY,
    '__module__' : 'detection_pb2'
    # @@protoc_insertion_point(class_scope:com.yimuzn.featuresdetection.device.grpc.DetectResponse.FeatureCountsEntry)
    })
  ,
  'DESCRIPTOR' : _DETECTRESPONSE,
  '__module__' : 'detection_pb2'
  # @@protoc_insertion_point(class_scope:com.yimuzn.featuresdetection.device.grpc.DetectResponse)
  })
_sym_db.RegisterMessage(DetectResponse)
_sym_db.RegisterMessage(DetectResponse.FeatureCountsEntry)


_DETECTRESPONSE_FEATURECOUNTSENTRY._options = None

_DETECTSERVICE = _descriptor.ServiceDescriptor(
  name='DetectService',
  full_name='com.yimuzn.featuresdetection.device.grpc.DetectService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=325,
  serialized_end=478,
  methods=[
  _descriptor.MethodDescriptor(
    name='DetectFeaturesYolo',
    full_name='com.yimuzn.featuresdetection.device.grpc.DetectService.DetectFeaturesYolo',
    index=0,
    containing_service=None,
    input_type=_DETECTREQUEST,
    output_type=_DETECTRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_DETECTSERVICE)

DESCRIPTOR.services_by_name['DetectService'] = _DETECTSERVICE

# @@protoc_insertion_point(module_scope)
