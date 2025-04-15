// RUN: mess-analysis --allow-unregistered-dialect %s count_generics | filecheck %s
// CHECK: 3
#map = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0) -> (d0)>
"builtin.module"() ({
  "func.func"() <{function_type = (tensor<4xf32>) -> tensor<3xf32>, sym_name = "linear_layer"}> ({
  ^bb0(%arg0: tensor<4xf32>):
    %0 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<4x3xf32>}> : () -> tensor<4x3xf32>
    %1 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %2 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<3xf32>}> : () -> tensor<3xf32>
    %3 = "tensor.expand_shape"(%arg0) <{reassociation = [[0, 1]], static_output_shape = array<i64: 1, 4>}> : (tensor<4xf32>) -> tensor<1x4xf32>
    %4 = "tensor.empty"() : () -> tensor<1x3xf32>
    %5 = "linalg.generic"(%1, %4) <{indexing_maps = [#map, #map1], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%arg7: f32, %arg8: f32):
      "linalg.yield"(%arg7) : (f32) -> ()
    }) : (f32, tensor<1x3xf32>) -> tensor<1x3xf32>
    %6 = "linalg.generic"(%3, %0, %5) <{indexing_maps = [#map2, #map3, #map4], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %11 = "arith.mulf"(%arg4, %arg5) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %12 = "arith.addf"(%arg6, %11) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%12) : (f32) -> ()
    }) : (tensor<1x4xf32>, tensor<4x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
    %7 = "tensor.collapse_shape"(%6) <{reassociation = [[0, 1]]}> : (tensor<1x3xf32>) -> tensor<3xf32>
    %8 = "tensor.empty"() : () -> tensor<3xf32>
    %9 = "linalg.generic"(%7, %2, %8) <{indexing_maps = [#map5, #map5, #map5], iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %10 = "arith.addf"(%arg1, %arg2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%10) : (f32) -> ()
    }) : (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    "func.return"(%9) : (tensor<3xf32>) -> ()
  }) : () -> ()
}) : () -> ()

