/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/tensor_scatter_update" />
import { Tensor } from '../tensor';
import { Rank, TensorLike } from '../types';
/**
 * Creates a new tensor by applying sparse updates to individual
 * values or slices to the passed in tensor according to
 * indices. This operator is the similar to scatterNd op, except that the
 * udpates are scattered on an existing tensor (as opposed to a zero-tensor).
 *
 * If indices contains duplicates, then we pick the last update for the index.
 *
 * If an out of bound index is found on CPU, an error is returned.
 *
 * Warning: There are some GPU specific semantics for this operation.
 *  - If an out of bound index is found, the index is ignored.
 *  - The order in which updates are applied is nondeterministic, so the output
 * will be nondeterministic if indices contains duplicates.
 * ```js
 * const shape = [8];
 * const tensor = tf.ones(shape);
 * const indices = tf.tensor2d([4, 3, 1, 7], [4, 1], 'int32');
 * const updates = tf.tensor1d([9, 10, 11, 12]);
 *
 * tf.tensorScatterUpdate(tensor, indices, updates).print();
 *    //[1, 11, 1, 10, 9, 1, 1, 12]
 * ```
 *
 * @param tensor A Tensor. Tensor to copy/update.
 * @param indices The tensor contains the indices into the output tensor, must
 *     have at least 2 axes: (num_updates, index_depth).
 * @param updates The tensor contains the value for the indices.
 *
 * @doc {heading: 'Operations', subheading: 'Slicing and Joining'}
 */
declare function tensorScatterUpdate_<R extends Rank>(tensor: Tensor<R> | TensorLike, indices: Tensor | TensorLike, updates: Tensor | TensorLike): Tensor<R>;
export declare const tensorScatterUpdate: typeof tensorScatterUpdate_;
export {};
