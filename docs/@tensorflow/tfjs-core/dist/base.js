/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
// base.ts is tfjs-core without auto registration of things like flags,
// gradients, chained ops or the opHandler. See base_side_effects.ts for parts
// tfjs core that are required side effects.
/**
 * @fileoverview
 * @suppress {partialAlias} Optimization disabled due to passing the module
 * object into a function below:
 *
 *   import * as ops from './ops/ops';
 *   setOpHandler(ops);
 */
// Serialization.
import * as io from './io/io';
import * as math from './math';
import * as broadcast_util from './ops/broadcast_util';
import * as browser from './ops/browser';
import * as gather_util from './ops/gather_nd_util';
import * as scatter_util from './ops/scatter_nd_util';
import * as slice_util from './ops/slice_util';
import * as serialization from './serialization';
import * as tensor_util from './tensor_util';
import * as test_util from './test_util';
import * as util from './util';
import { version } from './version';
export { AdadeltaOptimizer } from './optimizers/adadelta_optimizer';
export { AdagradOptimizer } from './optimizers/adagrad_optimizer';
export { AdamOptimizer } from './optimizers/adam_optimizer';
export { AdamaxOptimizer } from './optimizers/adamax_optimizer';
export { MomentumOptimizer } from './optimizers/momentum_optimizer';
export { Optimizer } from './optimizers/optimizer';
// Optimizers.
export { OptimizerConstructors } from './optimizers/optimizer_constructors';
export { RMSPropOptimizer } from './optimizers/rmsprop_optimizer';
export { SGDOptimizer } from './optimizers/sgd_optimizer';
export { Tensor, TensorBuffer, Variable } from './tensor';
export { Rank, sumOutType, upcastType } from './types';
export * from './ops/ops';
export { Reduction } from './ops/loss_ops_utils';
export * from './train';
export * from './globals';
export * from './kernel_registry';
export { customGrad, grad, grads, valueAndGrad, valueAndGrads, variableGrads } from './gradients';
export { Environment, env, ENV } from './environment';
export { version as version_core };
// Top-level method exports.
export { nextFrame } from './browser_util';
// Second level exports.
import * as backend_util from './backends/backend_util';
import * as device_util from './device_util';
export { browser, io, math, serialization, test_util, util, backend_util, broadcast_util, tensor_util, slice_util, gather_util, scatter_util, device_util };
import * as kernel_impls from './backends/kernel_impls';
export { kernel_impls };
// Backend specific.
export { KernelBackend, DataStorage } from './backends/backend';
// Export all kernel names / info.
export * from './kernel_names';
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYmFzZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvYmFzZS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCx1RUFBdUU7QUFDdkUsOEVBQThFO0FBQzlFLDRDQUE0QztBQUU1Qzs7Ozs7OztHQU9HO0FBRUgsaUJBQWlCO0FBQ2pCLE9BQU8sS0FBSyxFQUFFLE1BQU0sU0FBUyxDQUFDO0FBQzlCLE9BQU8sS0FBSyxJQUFJLE1BQU0sUUFBUSxDQUFDO0FBQy9CLE9BQU8sS0FBSyxjQUFjLE1BQU0sc0JBQXNCLENBQUM7QUFDdkQsT0FBTyxLQUFLLE9BQU8sTUFBTSxlQUFlLENBQUM7QUFDekMsT0FBTyxLQUFLLFdBQVcsTUFBTSxzQkFBc0IsQ0FBQztBQUNwRCxPQUFPLEtBQUssWUFBWSxNQUFNLHVCQUF1QixDQUFDO0FBQ3RELE9BQU8sS0FBSyxVQUFVLE1BQU0sa0JBQWtCLENBQUM7QUFDL0MsT0FBTyxLQUFLLGFBQWEsTUFBTSxpQkFBaUIsQ0FBQztBQUNqRCxPQUFPLEtBQUssV0FBVyxNQUFNLGVBQWUsQ0FBQztBQUM3QyxPQUFPLEtBQUssU0FBUyxNQUFNLGFBQWEsQ0FBQztBQUN6QyxPQUFPLEtBQUssSUFBSSxNQUFNLFFBQVEsQ0FBQztBQUMvQixPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBR2xDLE9BQU8sRUFBQyxpQkFBaUIsRUFBQyxNQUFNLGlDQUFpQyxDQUFDO0FBQ2xFLE9BQU8sRUFBQyxnQkFBZ0IsRUFBQyxNQUFNLGdDQUFnQyxDQUFDO0FBQ2hFLE9BQU8sRUFBQyxhQUFhLEVBQUMsTUFBTSw2QkFBNkIsQ0FBQztBQUMxRCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sK0JBQStCLENBQUM7QUFDOUQsT0FBTyxFQUFDLGlCQUFpQixFQUFDLE1BQU0saUNBQWlDLENBQUM7QUFDbEUsT0FBTyxFQUFDLFNBQVMsRUFBQyxNQUFNLHdCQUF3QixDQUFDO0FBQ2pELGNBQWM7QUFDZCxPQUFPLEVBQUMscUJBQXFCLEVBQUMsTUFBTSxxQ0FBcUMsQ0FBQztBQUMxRSxPQUFPLEVBQUMsZ0JBQWdCLEVBQUMsTUFBTSxnQ0FBZ0MsQ0FBQztBQUNoRSxPQUFPLEVBQUMsWUFBWSxFQUFDLE1BQU0sNEJBQTRCLENBQUM7QUFDeEQsT0FBTyxFQUEwRCxNQUFNLEVBQW9ELFlBQVksRUFBRSxRQUFRLEVBQUMsTUFBTSxVQUFVLENBQUM7QUFFbkssT0FBTyxFQUE0RixJQUFJLEVBQXdDLFVBQVUsRUFBMEIsVUFBVSxFQUF3QixNQUFNLFNBQVMsQ0FBQztBQUVyTyxjQUFjLFdBQVcsQ0FBQztBQUMxQixPQUFPLEVBQUMsU0FBUyxFQUFDLE1BQU0sc0JBQXNCLENBQUM7QUFFL0MsY0FBYyxTQUFTLENBQUM7QUFDeEIsY0FBYyxXQUFXLENBQUM7QUFDMUIsY0FBYyxtQkFBbUIsQ0FBQztBQUVsQyxPQUFPLEVBQUMsVUFBVSxFQUFFLElBQUksRUFBRSxLQUFLLEVBQUUsWUFBWSxFQUFFLGFBQWEsRUFBRSxhQUFhLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFHaEcsT0FBTyxFQUFDLFdBQVcsRUFBRSxHQUFHLEVBQUUsR0FBRyxFQUFDLE1BQU0sZUFBZSxDQUFDO0FBR3BELE9BQU8sRUFBQyxPQUFPLElBQUksWUFBWSxFQUFDLENBQUM7QUFFakMsNEJBQTRCO0FBQzVCLE9BQU8sRUFBQyxTQUFTLEVBQUMsTUFBTSxnQkFBZ0IsQ0FBQztBQUV6Qyx3QkFBd0I7QUFDeEIsT0FBTyxLQUFLLFlBQVksTUFBTSx5QkFBeUIsQ0FBQztBQUN4RCxPQUFPLEtBQUssV0FBVyxNQUFNLGVBQWUsQ0FBQztBQUM3QyxPQUFPLEVBQ0wsT0FBTyxFQUNQLEVBQUUsRUFDRixJQUFJLEVBQ0osYUFBYSxFQUNiLFNBQVMsRUFDVCxJQUFJLEVBQ0osWUFBWSxFQUNaLGNBQWMsRUFDZCxXQUFXLEVBQ1gsVUFBVSxFQUNWLFdBQVcsRUFDWCxZQUFZLEVBQ1osV0FBVyxFQUNaLENBQUM7QUFFRixPQUFPLEtBQUssWUFBWSxNQUFNLHlCQUF5QixDQUFDO0FBQ3hELE9BQU8sRUFBQyxZQUFZLEVBQUMsQ0FBQztBQUN0QixvQkFBb0I7QUFDcEIsT0FBTyxFQUFDLGFBQWEsRUFBZ0MsV0FBVyxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFFNUYsa0NBQWtDO0FBQ2xDLGNBQWMsZ0JBQWdCLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbi8vIGJhc2UudHMgaXMgdGZqcy1jb3JlIHdpdGhvdXQgYXV0byByZWdpc3RyYXRpb24gb2YgdGhpbmdzIGxpa2UgZmxhZ3MsXG4vLyBncmFkaWVudHMsIGNoYWluZWQgb3BzIG9yIHRoZSBvcEhhbmRsZXIuIFNlZSBiYXNlX3NpZGVfZWZmZWN0cy50cyBmb3IgcGFydHNcbi8vIHRmanMgY29yZSB0aGF0IGFyZSByZXF1aXJlZCBzaWRlIGVmZmVjdHMuXG5cbi8qKlxuICogQGZpbGVvdmVydmlld1xuICogQHN1cHByZXNzIHtwYXJ0aWFsQWxpYXN9IE9wdGltaXphdGlvbiBkaXNhYmxlZCBkdWUgdG8gcGFzc2luZyB0aGUgbW9kdWxlXG4gKiBvYmplY3QgaW50byBhIGZ1bmN0aW9uIGJlbG93OlxuICpcbiAqICAgaW1wb3J0ICogYXMgb3BzIGZyb20gJy4vb3BzL29wcyc7XG4gKiAgIHNldE9wSGFuZGxlcihvcHMpO1xuICovXG5cbi8vIFNlcmlhbGl6YXRpb24uXG5pbXBvcnQgKiBhcyBpbyBmcm9tICcuL2lvL2lvJztcbmltcG9ydCAqIGFzIG1hdGggZnJvbSAnLi9tYXRoJztcbmltcG9ydCAqIGFzIGJyb2FkY2FzdF91dGlsIGZyb20gJy4vb3BzL2Jyb2FkY2FzdF91dGlsJztcbmltcG9ydCAqIGFzIGJyb3dzZXIgZnJvbSAnLi9vcHMvYnJvd3Nlcic7XG5pbXBvcnQgKiBhcyBnYXRoZXJfdXRpbCBmcm9tICcuL29wcy9nYXRoZXJfbmRfdXRpbCc7XG5pbXBvcnQgKiBhcyBzY2F0dGVyX3V0aWwgZnJvbSAnLi9vcHMvc2NhdHRlcl9uZF91dGlsJztcbmltcG9ydCAqIGFzIHNsaWNlX3V0aWwgZnJvbSAnLi9vcHMvc2xpY2VfdXRpbCc7XG5pbXBvcnQgKiBhcyBzZXJpYWxpemF0aW9uIGZyb20gJy4vc2VyaWFsaXphdGlvbic7XG5pbXBvcnQgKiBhcyB0ZW5zb3JfdXRpbCBmcm9tICcuL3RlbnNvcl91dGlsJztcbmltcG9ydCAqIGFzIHRlc3RfdXRpbCBmcm9tICcuL3Rlc3RfdXRpbCc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4vdXRpbCc7XG5pbXBvcnQge3ZlcnNpb259IGZyb20gJy4vdmVyc2lvbic7XG5cbmV4cG9ydCB7SW5mZXJlbmNlTW9kZWwsIE1ldGFHcmFwaCwgTWV0YUdyYXBoSW5mbywgTW9kZWxQcmVkaWN0Q29uZmlnLCBNb2RlbFRlbnNvckluZm8sIFNhdmVkTW9kZWxUZW5zb3JJbmZvLCBTaWduYXR1cmVEZWYsIFNpZ25hdHVyZURlZkVudHJ5LCBTaWduYXR1cmVEZWZJbmZvfSBmcm9tICcuL21vZGVsX3R5cGVzJztcbmV4cG9ydCB7QWRhZGVsdGFPcHRpbWl6ZXJ9IGZyb20gJy4vb3B0aW1pemVycy9hZGFkZWx0YV9vcHRpbWl6ZXInO1xuZXhwb3J0IHtBZGFncmFkT3B0aW1pemVyfSBmcm9tICcuL29wdGltaXplcnMvYWRhZ3JhZF9vcHRpbWl6ZXInO1xuZXhwb3J0IHtBZGFtT3B0aW1pemVyfSBmcm9tICcuL29wdGltaXplcnMvYWRhbV9vcHRpbWl6ZXInO1xuZXhwb3J0IHtBZGFtYXhPcHRpbWl6ZXJ9IGZyb20gJy4vb3B0aW1pemVycy9hZGFtYXhfb3B0aW1pemVyJztcbmV4cG9ydCB7TW9tZW50dW1PcHRpbWl6ZXJ9IGZyb20gJy4vb3B0aW1pemVycy9tb21lbnR1bV9vcHRpbWl6ZXInO1xuZXhwb3J0IHtPcHRpbWl6ZXJ9IGZyb20gJy4vb3B0aW1pemVycy9vcHRpbWl6ZXInO1xuLy8gT3B0aW1pemVycy5cbmV4cG9ydCB7T3B0aW1pemVyQ29uc3RydWN0b3JzfSBmcm9tICcuL29wdGltaXplcnMvb3B0aW1pemVyX2NvbnN0cnVjdG9ycyc7XG5leHBvcnQge1JNU1Byb3BPcHRpbWl6ZXJ9IGZyb20gJy4vb3B0aW1pemVycy9ybXNwcm9wX29wdGltaXplcic7XG5leHBvcnQge1NHRE9wdGltaXplcn0gZnJvbSAnLi9vcHRpbWl6ZXJzL3NnZF9vcHRpbWl6ZXInO1xuZXhwb3J0IHtEYXRhVG9HUFVPcHRpb25zLCBEYXRhVG9HUFVXZWJHTE9wdGlvbiwgR1BVRGF0YSwgU2NhbGFyLCBUZW5zb3IsIFRlbnNvcjFELCBUZW5zb3IyRCwgVGVuc29yM0QsIFRlbnNvcjRELCBUZW5zb3I1RCwgVGVuc29yQnVmZmVyLCBWYXJpYWJsZX0gZnJvbSAnLi90ZW5zb3InO1xuZXhwb3J0IHtHcmFkU2F2ZUZ1bmMsIE5hbWVkVGVuc29yTWFwLCBUZW5zb3JDb250YWluZXIsIFRlbnNvckNvbnRhaW5lckFycmF5LCBUZW5zb3JDb250YWluZXJPYmplY3R9IGZyb20gJy4vdGVuc29yX3R5cGVzJztcbmV4cG9ydCB7QmFja2VuZFZhbHVlcywgRGF0YVR5cGUsIERhdGFUeXBlTWFwLCBEYXRhVHlwZUZvciwgRGF0YVZhbHVlcywgTnVtZXJpY0RhdGFUeXBlLCBQaXhlbERhdGEsIFJhbmssIFJlY3Vyc2l2ZUFycmF5LCBTY2FsYXJMaWtlLCBTaGFwZU1hcCwgc3VtT3V0VHlwZSwgVGVuc29yTGlrZSwgVHlwZWRBcnJheSwgdXBjYXN0VHlwZSwgV2ViR0xEYXRhLCBXZWJHUFVEYXRhfSBmcm9tICcuL3R5cGVzJztcblxuZXhwb3J0ICogZnJvbSAnLi9vcHMvb3BzJztcbmV4cG9ydCB7UmVkdWN0aW9ufSBmcm9tICcuL29wcy9sb3NzX29wc191dGlscyc7XG5cbmV4cG9ydCAqIGZyb20gJy4vdHJhaW4nO1xuZXhwb3J0ICogZnJvbSAnLi9nbG9iYWxzJztcbmV4cG9ydCAqIGZyb20gJy4va2VybmVsX3JlZ2lzdHJ5JztcbmV4cG9ydCB7VGVuc29ySW5mbywgRGF0YUlkfSBmcm9tICcuL3RlbnNvcl9pbmZvJztcbmV4cG9ydCB7Y3VzdG9tR3JhZCwgZ3JhZCwgZ3JhZHMsIHZhbHVlQW5kR3JhZCwgdmFsdWVBbmRHcmFkcywgdmFyaWFibGVHcmFkc30gZnJvbSAnLi9ncmFkaWVudHMnO1xuXG5leHBvcnQge1RpbWluZ0luZm8sIE1lbW9yeUluZm8sIEZvcndhcmRGdW5jfSBmcm9tICcuL2VuZ2luZSc7XG5leHBvcnQge0Vudmlyb25tZW50LCBlbnYsIEVOVn0gZnJvbSAnLi9lbnZpcm9ubWVudCc7XG5leHBvcnQge1BsYXRmb3JtfSBmcm9tICcuL3BsYXRmb3Jtcy9wbGF0Zm9ybSc7XG5cbmV4cG9ydCB7dmVyc2lvbiBhcyB2ZXJzaW9uX2NvcmV9O1xuXG4vLyBUb3AtbGV2ZWwgbWV0aG9kIGV4cG9ydHMuXG5leHBvcnQge25leHRGcmFtZX0gZnJvbSAnLi9icm93c2VyX3V0aWwnO1xuXG4vLyBTZWNvbmQgbGV2ZWwgZXhwb3J0cy5cbmltcG9ydCAqIGFzIGJhY2tlbmRfdXRpbCBmcm9tICcuL2JhY2tlbmRzL2JhY2tlbmRfdXRpbCc7XG5pbXBvcnQgKiBhcyBkZXZpY2VfdXRpbCBmcm9tICcuL2RldmljZV91dGlsJztcbmV4cG9ydCB7XG4gIGJyb3dzZXIsXG4gIGlvLFxuICBtYXRoLFxuICBzZXJpYWxpemF0aW9uLFxuICB0ZXN0X3V0aWwsXG4gIHV0aWwsXG4gIGJhY2tlbmRfdXRpbCxcbiAgYnJvYWRjYXN0X3V0aWwsXG4gIHRlbnNvcl91dGlsLFxuICBzbGljZV91dGlsLFxuICBnYXRoZXJfdXRpbCxcbiAgc2NhdHRlcl91dGlsLFxuICBkZXZpY2VfdXRpbFxufTtcblxuaW1wb3J0ICogYXMga2VybmVsX2ltcGxzIGZyb20gJy4vYmFja2VuZHMva2VybmVsX2ltcGxzJztcbmV4cG9ydCB7a2VybmVsX2ltcGxzfTtcbi8vIEJhY2tlbmQgc3BlY2lmaWMuXG5leHBvcnQge0tlcm5lbEJhY2tlbmQsIEJhY2tlbmRUaW1pbmdJbmZvLCBEYXRhTW92ZXIsIERhdGFTdG9yYWdlfSBmcm9tICcuL2JhY2tlbmRzL2JhY2tlbmQnO1xuXG4vLyBFeHBvcnQgYWxsIGtlcm5lbCBuYW1lcyAvIGluZm8uXG5leHBvcnQgKiBmcm9tICcuL2tlcm5lbF9uYW1lcyc7XG4iXX0=