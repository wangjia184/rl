/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/random_uniform" />
import { Tensor } from '../tensor';
import { DataType, Rank, ShapeMap } from '../types';
/**
 * Creates a `tf.Tensor` with values sampled from a uniform distribution.
 *
 * The generated values follow a uniform distribution in the range [minval,
 * maxval). The lower bound minval is included in the range, while the upper
 * bound maxval is excluded.
 *
 * ```js
 * tf.randomUniform([2, 2]).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param minval The lower bound on the range of random values to generate.
 *   Defaults to 0.
 * @param maxval The upper bound on the range of random values to generate.
 *   Defaults to 1.
 * @param dtype The data type of the output tensor. Defaults to 'float32'.
 * @param seed An optional int. Defaults to 0. If seed is set to be non-zero,
 *   the random number generator is seeded by the given seed. Otherwise, it is
 *   seeded by a random seed.
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
declare function randomUniform_<R extends Rank>(shape: ShapeMap[R], minval?: number, maxval?: number, dtype?: DataType, seed?: number | string): Tensor<R>;
export declare const randomUniform: typeof randomUniform_;
export {};
