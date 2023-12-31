/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/layers/convolutional_depthwise_serialization" />
import { ConstraintSerialization } from '../constraint_config';
import { InitializerSerialization } from '../initializer_config';
import { RegularizerSerialization } from '../regularizer_config';
import { BaseLayerSerialization } from '../topology_config';
import { BaseConvLayerConfig } from './convolutional_serialization';
export interface DepthwiseConv2DLayerConfig extends BaseConvLayerConfig {
    kernel_size: number | [number, number];
    depth_multiplier?: number;
    depthwise_initializer?: InitializerSerialization;
    depthwise_constraint?: ConstraintSerialization;
    depthwise_regularizer?: RegularizerSerialization;
}
export type DepthwiseConv2DLayerSerialization = BaseLayerSerialization<'DepthwiseConv2D', DepthwiseConv2DLayerConfig>;
export type ConvolutionalDepthwiseLayerSerialization = DepthwiseConv2DLayerSerialization;
export type ConvolutionalDepthwiseLayerClassName = ConvolutionalDepthwiseLayerSerialization['class_name'];
/**
 * A string array of valid ConvolutionalDepthwiseLayer class names.
 *
 * This is guaranteed to match the `ConvolutionalDepthwiseLayerClassName` union
 * type.
 */
export declare const convolutionalDepthwiseLayerClassNames: ConvolutionalDepthwiseLayerClassName[];
