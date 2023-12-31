/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/common" />
/** @docalias (null | number)[] */
export type Shape = Array<null | number>;
export type DataType = 'float32' | 'int32' | 'bool' | 'complex64' | 'string';
/** @docinline */
export type DataFormat = 'channelsFirst' | 'channelsLast';
export declare const VALID_DATA_FORMAT_VALUES: string[];
export type InterpolationFormat = 'nearest' | 'bilinear';
export declare const VALID_INTERPOLATION_FORMAT_VALUES: string[];
export type DataFormatSerialization = 'channels_first' | 'channels_last';
/** @docinline */
export type PaddingMode = 'valid' | 'same' | 'causal';
export declare const VALID_PADDING_MODE_VALUES: string[];
/** @docinline */
export type PoolMode = 'max' | 'avg';
export declare const VALID_POOL_MODE_VALUES: string[];
/** @docinline */
export type BidirectionalMergeMode = 'sum' | 'mul' | 'concat' | 'ave';
export declare const VALID_BIDIRECTIONAL_MERGE_MODES: string[];
/** @docinline */
export type SampleWeightMode = 'temporal';
export declare const VALID_SAMPLE_WEIGHT_MODES: string[];
