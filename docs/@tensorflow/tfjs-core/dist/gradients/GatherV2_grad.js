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
import { GatherV2 } from '../kernel_names';
import { getUndoAxesPermutation } from '../ops/axis_util';
import { reshape } from '../ops/reshape';
import { stack } from '../ops/stack';
import { transpose } from '../ops/transpose';
import { unsortedSegmentSum } from '../ops/unsorted_segment_sum';
import { parseAxisParam } from '../util';
export const gatherGradConfig = {
    kernelName: GatherV2,
    inputsToSave: ['x', 'indices'],
    gradFunc: (dy, saved, attrs) => {
        const [x, indices] = saved;
        const { axis, batchDims } = attrs;
        const parsedAxis = parseAxisParam(axis, x.shape)[0];
        const derXBatch = (x, indices, dy) => {
            return () => {
                const paramsShape = x.shape;
                const indicesSize = indices.size;
                const outerShape = paramsShape.slice(0, parsedAxis);
                const outerDims = outerShape.length;
                const innerShape = paramsShape.slice(axis, paramsShape.length).slice(1);
                const innerDims = innerShape.length;
                const outerAxesIndices = arrayRange(0, outerDims);
                const innerAxesIndices = arrayRange(outerDims + 1, outerDims + 1 + innerDims);
                const valuesShape = arrayConcat([outerShape, [indicesSize],
                    innerShape]);
                const values = reshape(dy, valuesShape);
                const reshapedIndices = reshape(indices, [indicesSize]);
                const transposeDims = arrayConcat([[outerDims], outerAxesIndices, innerAxesIndices]);
                const valuesTranspose = transpose(values, transposeDims);
                let paramsGrad = unsortedSegmentSum(valuesTranspose, reshapedIndices, x.shape[parsedAxis]);
                const invertTransposeDims = getUndoAxesPermutation(transposeDims);
                paramsGrad = transpose(paramsGrad, invertTransposeDims);
                return paramsGrad;
            };
        };
        if (batchDims === 1) {
            const batchSize = x.shape[0];
            const xBatch = x.split(batchSize, 0);
            const derXBatched = () => {
                const stacked = stack(xBatch.map((x, i) => {
                    return derXBatch(x, indices.slice(i, 1), dy.slice(i, 1))();
                }));
                return stacked.reshape(x.shape);
            };
            return { x: derXBatched, indices: () => indices };
        }
        else {
            return { x: derXBatch(x, indices, dy), indices: () => indices };
        }
    }
};
function arrayRange(start, stop) {
    const result = [];
    for (let i = start; i < stop; ++i) {
        result.push(i);
    }
    return result;
}
function arrayConcat(arrays) {
    const result = [];
    for (let i = 0; i < arrays.length; ++i) {
        for (let j = 0; j < arrays[i].length; ++j) {
            result.push(arrays[i][j]);
        }
    }
    return result;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiR2F0aGVyVjJfZ3JhZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvZ3JhZGllbnRzL0dhdGhlclYyX2dyYWQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLFFBQVEsRUFBZ0IsTUFBTSxpQkFBaUIsQ0FBQztBQUV4RCxPQUFPLEVBQUMsc0JBQXNCLEVBQUMsTUFBTSxrQkFBa0IsQ0FBQztBQUN4RCxPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sZ0JBQWdCLENBQUM7QUFDdkMsT0FBTyxFQUFDLEtBQUssRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUNuQyxPQUFPLEVBQUMsU0FBUyxFQUFDLE1BQU0sa0JBQWtCLENBQUM7QUFDM0MsT0FBTyxFQUFDLGtCQUFrQixFQUFDLE1BQU0sNkJBQTZCLENBQUM7QUFFL0QsT0FBTyxFQUFDLGNBQWMsRUFBQyxNQUFNLFNBQVMsQ0FBQztBQUV2QyxNQUFNLENBQUMsTUFBTSxnQkFBZ0IsR0FBZTtJQUMxQyxVQUFVLEVBQUUsUUFBUTtJQUNwQixZQUFZLEVBQUUsQ0FBQyxHQUFHLEVBQUUsU0FBUyxDQUFDO0lBQzlCLFFBQVEsRUFBRSxDQUFDLEVBQVUsRUFBRSxLQUFlLEVBQUUsS0FBbUIsRUFBRSxFQUFFO1FBQzdELE1BQU0sQ0FBQyxDQUFDLEVBQUUsT0FBTyxDQUFDLEdBQUcsS0FBSyxDQUFDO1FBQzNCLE1BQU0sRUFBQyxJQUFJLEVBQUUsU0FBUyxFQUFDLEdBQUcsS0FBaUMsQ0FBQztRQUU1RCxNQUFNLFVBQVUsR0FBRyxjQUFjLENBQUMsSUFBSSxFQUFFLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUVwRCxNQUFNLFNBQVMsR0FBRyxDQUFDLENBQVMsRUFBRSxPQUFlLEVBQUUsRUFBVSxFQUFFLEVBQUU7WUFDM0QsT0FBTyxHQUFXLEVBQUU7Z0JBQ2xC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QUFDO0NBQ0YsQ0FBQztBQUVGLFNBQVMsVUFBVSxDQUFDLEtBQWEsRUFBRSxJQUFZO0lBQzdDLE1BQU0sTUFBTSxHQUFHLEVBQUUsQ0FBQztJQUNsQixLQUFLLElBQUksQ0FBQyxHQUFHLEtBQUssRUFBRSxDQUFDLEdBQUcsSUFBSSxFQUFFLEVBQUUsQ0FBQyxFQUFFO1FBQ2pDLE1BQU0sQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7S0FDaEI7SUFDRCxPQUFPLE1BQU0sQ0FBQztBQUNoQixDQUFDO0FBRUQsU0FBUyxXQUFXLENBQUMsTUFBa0I7SUFDckMsTUFBTSxNQUFNLEdBQUcsRUFBRSxDQUFDO0lBQ2xCLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxNQUFNLENBQUMsTUFBTSxFQUFFLEVBQUUsQ0FBQyxFQUFFO1FBQ3RDLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsTUFBTSxFQUFFLEVBQUUsQ0FBQyxFQUFFO1lBQ3pDLE1BQU0sQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7U0FDM0I7S0FDRjtJQUNELE9BQU8sTUFBTSxDQUFDO0FBQ2hCLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7R2F0aGVyVjIsIEdhdGhlclYyQXR0cnN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge0dyYWRDb25maWcsIE5hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7Z2V0VW5kb0F4ZXNQZXJtdXRhdGlvbn0gZnJvbSAnLi4vb3BzL2F4aXNfdXRpbCc7XG5pbXBvcnQge3Jlc2hhcGV9IGZyb20gJy4uL29wcy9yZXNoYXBlJztcbmltcG9ydCB7c3RhY2t9IGZyb20gJy4uL29wcy9zdGFjayc7XG5pbXBvcnQge3RyYW5zcG9zZX0gZnJvbSAnLi4vb3BzL3RyYW5zcG9zZSc7XG5pbXBvcnQge3Vuc29ydGVkU2VnbWVudFN1bX0gZnJvbSAnLi4vb3BzL3Vuc29ydGVkX3NlZ21lbnRfc3VtJztcbmltcG9ydCB7VGVuc29yLCBUZW5zb3IxRH0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7cGFyc2VBeGlzUGFyYW19IGZyb20gJy4uL3V0aWwnO1xuXG5leHBvcnQgY29uc3QgZ2F0aGVyR3JhZENvbmZpZzogR3JhZENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogR2F0aGVyVjIsXG4gIGlucHV0c1RvU2F2ZTogWyd4JywgJ2luZGljZXMnXSxcbiAgZ3JhZEZ1bmM6IChkeTogVGVuc29yLCBzYXZlZDogVGVuc29yW10sIGF0dHJzOiBOYW1lZEF0dHJNYXApID0+IHtcbiAgICBjb25zdCBbeCwgaW5kaWNlc10gPSBzYXZlZDtcbiAgICBjb25zdCB7YXhpcywgYmF0Y2hEaW1zfSA9IGF0dHJzIGFzIHVua25vd24gYXMgR2F0aGVyVjJBdHRycztcblxuICAgIGNvbnN0IHBhcnNlZEF4aXMgPSBwYXJzZUF4aXNQYXJhbShheGlzLCB4LnNoYXBlKVswXTtcblxuICAgIGNvbnN0IGRlclhCYXRjaCA9ICh4OiBUZW5zb3IsIGluZGljZXM6IFRlbnNvciwgZHk6IFRlbnNvcikgPT4ge1xuICAgICAgcmV0dXJuICgpOiBUZW5zb3IgPT4ge1xuICAgICAgICBjb25zdCBwYXJhbXNTaGFwZSA9IHguc2hhcGU7XG4gICAgICAgIGNvbnN0IGluZGljZXNTaXplID0gaW5kaWNlcy5zaXplO1xuXG4gICAgICAgIGNvbnN0IG91dGVyU2hhcGUgPSBwYXJhbXNTaGFwZS5zbGljZSgwLCBwYXJzZWRBeGlzKTtcbiAgICAgICAgY29uc3Qgb3V0ZXJEaW1zID0gb3V0ZXJTaGFwZS5sZW5ndGg7XG4gICAgICAgIGNvbnN0IGlubmVyU2hhcGUgPSBwYXJhbXNTaGFwZS5zbGljZShheGlzLCBwYXJhbXNTaGFwZS5sZW5ndGgpLnNsaWNlKDEpO1xuICAgICAgICBjb25zdCBpbm5lckRpbXMgPSBpbm5lclNoYXBlLmxlbmd0aDtcblxuICAgICAgICBjb25zdCBvdXRlckF4ZXNJbmRpY2VzID0gYXJyYXlSYW5nZSgwLCBvdXRlckRpbXMpO1xuICAgICAgICBjb25zdCBpbm5lckF4ZXNJbmRpY2VzID1cbiAgICAgICAgICAgIGFycmF5UmFuZ2Uob3V0ZXJEaW1zICsgMSwgb3V0ZXJEaW1zICsgMSArIGlubmVyRGltcyk7XG5cbiAgICAgICAgY29uc3QgdmFsdWVzU2hhcGUgPSBhcnJheUNvbmNhdChbb3V0ZXJTaGFwZSwgW2luZGljZXNTaXplXSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgaW5uZXJTaGFwZV0pO1xuXG4gICAgICAgIGNvbnN0IHZhbHVlcyA9IHJlc2hhcGUoZHksIHZhbHVlc1NoYXBlKTtcbiAgICAgICAgY29uc3QgcmVzaGFwZWRJbmRpY2VzID0gcmVzaGFwZShpbmRpY2VzLCBbaW5kaWNlc1NpemVdKTtcblxuICAgICAgICBjb25zdCB0cmFuc3Bvc2VEaW1zID1cbiAgICAgICAgICAgIGFycmF5Q29uY2F0KFtbb3V0ZXJEaW1zXSwgb3V0ZXJBeGVzSW5kaWNlcywgaW5uZXJBeGVzSW5kaWNlc10pO1xuICAgICAgICBjb25zdCB2YWx1ZXNUcmFuc3Bvc2UgPSB0cmFuc3Bvc2UodmFsdWVzLCB0cmFuc3Bvc2VEaW1zKTtcbiAgICAgICAgbGV0IHBhcmFtc0dyYWQgPSB1bnNvcnRlZFNlZ21lbnRTdW0oXG4gICAgICAgICAgICB2YWx1ZXNUcmFuc3Bvc2UsIHJlc2hhcGVkSW5kaWNlcyBhcyBUZW5zb3IxRCwgeC5zaGFwZVtwYXJzZWRBeGlzXSk7XG4gICAgICAgIGNvbnN0IGludmVydFRyYW5zcG9zZURpbXMgPSBnZXRVbmRvQXhlc1Blcm11dGF0aW9uKHRyYW5zcG9zZURpbXMpO1xuICAgICAgICBwYXJhbXNHcmFkID0gdHJhbnNwb3NlKHBhcmFtc0dyYWQsIGludmVydFRyYW5zcG9zZURpbXMpO1xuICAgICAgICByZXR1cm4gcGFyYW1zR3JhZDtcbiAgICAgIH07XG4gICAgfTtcblxuICAgIGlmIChiYXRjaERpbXMgPT09IDEpIHtcbiAgICAgIGNvbnN0IGJhdGNoU2l6ZSA9IHguc2hhcGVbMF07XG4gICAgICBjb25zdCB4QmF0Y2ggPSB4LnNwbGl0KGJhdGNoU2l6ZSwgMCk7XG4gICAgICBjb25zdCBkZXJYQmF0Y2hlZCA9ICgpID0+IHtcbiAgICAgICAgY29uc3Qgc3RhY2tlZCA9IHN0YWNrKFxuICAgICAgICAgIHhCYXRjaC5tYXAoKHgsIGkpID0+IHtcbiAgICAgICAgICAgIHJldHVybiBkZXJYQmF0Y2goeCwgaW5kaWNlcy5zbGljZShpLDEpLCBkeS5zbGljZShpLDEpKSgpO1xuICAgICAgICAgIH0pKTtcbiAgICAgICAgcmV0dXJuIHN0YWNrZWQucmVzaGFwZSh4LnNoYXBlKTtcbiAgICAgIH07XG4gICAgICByZXR1cm4ge3g6IGRlclhCYXRjaGVkLCBpbmRpY2VzOiAoKSA9PiBpbmRpY2VzfTtcbiAgICB9IGVsc2Uge1xuICAgICAgcmV0dXJuIHt4OiBkZXJYQmF0Y2goeCwgaW5kaWNlcywgZHkpLCBpbmRpY2VzOiAoKSA9PiBpbmRpY2VzfTtcbiAgICB9XG4gIH1cbn07XG5cbmZ1bmN0aW9uIGFycmF5UmFuZ2Uoc3RhcnQ6IG51bWJlciwgc3RvcDogbnVtYmVyKTogbnVtYmVyW10ge1xuICBjb25zdCByZXN1bHQgPSBbXTtcbiAgZm9yIChsZXQgaSA9IHN0YXJ0OyBpIDwgc3RvcDsgKytpKSB7XG4gICAgcmVzdWx0LnB1c2goaSk7XG4gIH1cbiAgcmV0dXJuIHJlc3VsdDtcbn1cblxuZnVuY3Rpb24gYXJyYXlDb25jYXQoYXJyYXlzOiBudW1iZXJbXVtdKTogbnVtYmVyW10ge1xuICBjb25zdCByZXN1bHQgPSBbXTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBhcnJheXMubGVuZ3RoOyArK2kpIHtcbiAgICBmb3IgKGxldCBqID0gMDsgaiA8IGFycmF5c1tpXS5sZW5ndGg7ICsraikge1xuICAgICAgcmVzdWx0LnB1c2goYXJyYXlzW2ldW2pdKTtcbiAgICB9XG4gIH1cbiAgcmV0dXJuIHJlc3VsdDtcbn1cbiJdfQ==