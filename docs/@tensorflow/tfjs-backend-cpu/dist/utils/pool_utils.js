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
import { buffer } from '@tensorflow/tfjs-core';
export function pool(xValues, xShape, dtype, strides, convInfo, poolType) {
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    const initialValue = (poolType === 'max' ? Number.NEGATIVE_INFINITY :
        Number.POSITIVE_INFINITY);
    const output = buffer(convInfo.outShape, dtype);
    const outputVals = output.values;
    const outputBatchStrides = convInfo.outShape[1] * convInfo.outShape[2] * convInfo.outShape[3];
    const outputRowStrides = convInfo.outShape[2] * convInfo.outShape[3];
    const outputColStrides = convInfo.outShape[3];
    for (let b = 0; b < convInfo.batchSize; ++b) {
        const outputBatchOffset = b * outputBatchStrides;
        const inputBatchOffset = b * strides[0];
        for (let d = 0; d < convInfo.inChannels; ++d) {
            for (let yR = 0; yR < convInfo.outHeight; ++yR) {
                const xRCorner = yR * strideHeight - padTop;
                const xRMin = Math.max(0, xRCorner);
                const xRMax = Math.min(convInfo.inHeight, effectiveFilterHeight + xRCorner);
                const outputRowOffset = outputBatchOffset + yR * outputRowStrides;
                for (let yC = 0; yC < convInfo.outWidth; ++yC) {
                    const xCCorner = yC * strideWidth - padLeft;
                    const xCMin = Math.max(0, xCCorner);
                    const xCMax = Math.min(convInfo.inWidth, effectiveFilterWidth + xCCorner);
                    let minMaxValue = initialValue;
                    let avgValue = 0;
                    let count = 0;
                    for (let xR = xRMin; xR < xRMax; xR += dilationHeight) {
                        const xROffset = inputBatchOffset + xR * strides[1];
                        for (let xC = xCMin; xC < xCMax; xC += dilationWidth) {
                            const xCOffset = xROffset + xC * strides[2];
                            const pixel = xValues[xCOffset + d];
                            if ((poolType === 'max' && pixel > minMaxValue)) {
                                minMaxValue = pixel;
                            }
                            else if (poolType === 'avg') {
                                avgValue += pixel;
                                count++;
                            }
                        }
                        if (isNaN(minMaxValue)) {
                            break;
                        }
                    }
                    const outputOffset = outputRowOffset + yC * outputColStrides + d;
                    outputVals[outputOffset] =
                        poolType === 'avg' ? avgValue / count : minMaxValue;
                }
            }
        }
    }
    return output;
}
export function maxPoolPositions(xValues, xShape, dtype, convInfo, flattenPositions = false, includeBatchInIndex = false) {
    const maxPositions = buffer(convInfo.outShape, 'int32');
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    const xBuf = buffer(xShape, dtype, xValues);
    for (let b = 0; b < convInfo.batchSize; ++b) {
        for (let d = 0; d < convInfo.inChannels; ++d) {
            for (let yR = 0; yR < convInfo.outHeight; ++yR) {
                const xRCorner = yR * strideHeight - padTop;
                let xRMin = xRCorner;
                while (xRMin < 0) {
                    xRMin += dilationHeight;
                }
                // const xRMin = Math.max(0, xRCorner);
                const xRMax = Math.min(convInfo.inHeight, effectiveFilterHeight + xRCorner);
                for (let yC = 0; yC < convInfo.outWidth; ++yC) {
                    const xCCorner = yC * strideWidth - padLeft;
                    let xCMin = xCCorner;
                    while (xCMin < 0) {
                        xCMin += dilationWidth;
                    }
                    const xCMax = Math.min(convInfo.inWidth, effectiveFilterWidth + xCCorner);
                    let maxValue = Number.NEGATIVE_INFINITY;
                    let maxPosition = -1;
                    for (let xR = xRMin; xR < xRMax; xR += dilationHeight) {
                        const wR = xR - xRCorner;
                        for (let xC = xCMin; xC < xCMax; xC += dilationWidth) {
                            const wC = xC - xCCorner;
                            // For some reason, disable-next-line is not working
                            // TODO(mattsoulanille): Remove this when switching to TS5.
                            /* tslint:disable: no-unnecessary-type-assertion */
                            const pixel = xBuf.get(b, xR, xC, d);
                            if (pixel > maxValue) {
                                maxValue = pixel;
                                if (flattenPositions) {
                                    maxPosition = includeBatchInIndex ?
                                        ((b * convInfo.inHeight + xR) * convInfo.inWidth + xC) *
                                            convInfo.inChannels +
                                            d :
                                        (xR * convInfo.inWidth + xC) * convInfo.inChannels + d;
                                }
                                else {
                                    maxPosition = wR * effectiveFilterWidth + wC;
                                }
                            }
                        }
                    }
                    maxPositions.set(maxPosition, b, yR, yC, d);
                }
            }
        }
    }
    return maxPositions;
}
export function pool3d(xValues, xShape, dtype, strides, convInfo, poolType) {
    const strideDepth = convInfo.strideDepth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationDepth = convInfo.dilationDepth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterDepth = convInfo.effectiveFilterDepth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padFront = convInfo.padInfo.front;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    const initialValue = (poolType === 'max' ? Number.NEGATIVE_INFINITY :
        Number.POSITIVE_INFINITY);
    const output = buffer(convInfo.outShape, dtype);
    const outputVals = output.values;
    const outputBatchStrides = convInfo.outShape[1] * convInfo.outShape[2] *
        convInfo.outShape[3] * convInfo.outShape[4];
    const outputDepthStrides = convInfo.outShape[2] * convInfo.outShape[3] * convInfo.outShape[4];
    const outputRowStrides = convInfo.outShape[3] * convInfo.outShape[4];
    const outputColStrides = convInfo.outShape[4];
    for (let batch = 0; batch < convInfo.batchSize; ++batch) {
        const outputBatchOffset = batch * outputBatchStrides;
        const inputBatchOffset = batch * strides[0];
        for (let channel = 0; channel < convInfo.inChannels; ++channel) {
            for (let yDepth = 0; yDepth < convInfo.outDepth; ++yDepth) {
                const xDepthCorner = yDepth * strideDepth - padFront;
                let xDepthMin = xDepthCorner;
                while (xDepthMin < 0) {
                    xDepthMin += dilationDepth;
                }
                const xDepthMax = Math.min(convInfo.inDepth, effectiveFilterDepth + xDepthCorner);
                const outputDepthOffset = outputBatchOffset + yDepth * outputDepthStrides;
                for (let yRow = 0; yRow < convInfo.outHeight; ++yRow) {
                    const xRowCorner = yRow * strideHeight - padTop;
                    let xRowMin = xRowCorner;
                    while (xRowMin < 0) {
                        xRowMin += dilationHeight;
                    }
                    const xRowMax = Math.min(convInfo.inHeight, effectiveFilterHeight + xRowCorner);
                    const outputRowOffset = outputDepthOffset + yRow * outputRowStrides;
                    for (let yCol = 0; yCol < convInfo.outWidth; ++yCol) {
                        const xColCorner = yCol * strideWidth - padLeft;
                        let xColMin = xColCorner;
                        while (xColMin < 0) {
                            xColMin += dilationWidth;
                        }
                        const xColMax = Math.min(convInfo.inWidth, effectiveFilterWidth + xColCorner);
                        // Shader code begins
                        const outputColOffset = outputRowOffset + yCol * outputColStrides;
                        let minMaxValue = initialValue;
                        let avgValue = 0;
                        let count = 0;
                        for (let xDepth = xDepthMin; xDepth < xDepthMax; xDepth += dilationDepth) {
                            const xDepthOffset = inputBatchOffset + xDepth * strides[1];
                            for (let xRow = xRowMin; xRow < xRowMax; xRow += dilationHeight) {
                                const xRowOffset = xDepthOffset + xRow * strides[2];
                                for (let xCol = xColMin; xCol < xColMax; xCol += dilationWidth) {
                                    const xColOffset = xRowOffset + xCol * strides[3];
                                    const pixel = xValues[xColOffset + channel];
                                    if ((poolType === 'max' && pixel > minMaxValue)) {
                                        minMaxValue = pixel;
                                    }
                                    else if (poolType === 'avg') {
                                        avgValue += pixel;
                                        count++;
                                    }
                                    if (isNaN(minMaxValue)) {
                                        break;
                                    }
                                }
                                if (isNaN(minMaxValue)) {
                                    break;
                                }
                            }
                            if (isNaN(minMaxValue)) {
                                break;
                            }
                        }
                        const outputOffset = outputColOffset + channel;
                        outputVals[outputOffset] = poolType === 'avg' ?
                            avgValue / Math.max(count, 1) :
                            minMaxValue;
                    }
                }
            }
        }
    }
    return output;
}
export function maxPool3dPositions(xBuf, convInfo) {
    const maxPositions = buffer(convInfo.outShape, 'int32');
    const strideDepth = convInfo.strideDepth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationDepth = convInfo.dilationDepth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterDepth = convInfo.effectiveFilterDepth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padFront = convInfo.padInfo.front;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    for (let batch = 0; batch < convInfo.batchSize; ++batch) {
        for (let channel = 0; channel < convInfo.inChannels; ++channel) {
            for (let yDepth = 0; yDepth < convInfo.outDepth; ++yDepth) {
                const xDepthCorner = yDepth * strideDepth - padFront;
                let xDepthMin = xDepthCorner;
                while (xDepthMin < 0) {
                    xDepthMin += dilationDepth;
                }
                const xDepthMax = Math.min(convInfo.inDepth, effectiveFilterDepth + xDepthCorner);
                for (let yRow = 0; yRow < convInfo.outHeight; ++yRow) {
                    const xRowCorner = yRow * strideHeight - padTop;
                    let xRowMin = xRowCorner;
                    while (xRowMin < 0) {
                        xRowMin += dilationHeight;
                    }
                    const xRowMax = Math.min(convInfo.inHeight, effectiveFilterHeight + xRowCorner);
                    for (let yCol = 0; yCol < convInfo.outWidth; ++yCol) {
                        const xColCorner = yCol * strideWidth - padLeft;
                        let xColMin = xColCorner;
                        while (xColMin < 0) {
                            xColMin += dilationWidth;
                        }
                        const xColMax = Math.min(convInfo.inWidth, effectiveFilterWidth + xColCorner);
                        // Shader code begins
                        let maxValue = Number.NEGATIVE_INFINITY;
                        let maxPosition = -1;
                        for (let xDepth = xDepthMin; xDepth < xDepthMax; xDepth += dilationDepth) {
                            const wDepth = xDepth - xDepthCorner;
                            for (let xRow = xRowMin; xRow < xRowMax; xRow += dilationHeight) {
                                const wRow = xRow - xRowCorner;
                                for (let xCol = xColMin; xCol < xColMax; xCol += dilationWidth) {
                                    const wCol = xCol - xColCorner;
                                    const pixel = xBuf.get(batch, xDepth, xRow, xCol, channel);
                                    if (pixel >= maxValue) {
                                        maxValue = pixel;
                                        maxPosition =
                                            wDepth * effectiveFilterHeight * effectiveFilterWidth +
                                                wRow * effectiveFilterHeight + wCol;
                                    }
                                }
                            }
                        }
                        maxPositions.set(maxPosition, batch, yDepth, yRow, yCol, channel);
                    }
                }
            }
        }
    }
    return maxPositions;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9vbF91dGlscy5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC1jcHUvc3JjL3V0aWxzL3Bvb2xfdXRpbHMudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFlLE1BQU0sRUFBMkMsTUFBTSx1QkFBdUIsQ0FBQztBQUVyRyxNQUFNLFVBQVUsSUFBSSxDQUNoQixPQUFtQixFQUFFLE1BQWdCLEVBQUUsS0FBZSxFQUFFLE9BQWlCLEVBQ3pFLFFBQWlDLEVBQ2pDLFFBQXFCO0lBQ3ZCLE1BQU0sWUFBWSxHQUFHLFFBQVEsQ0FBQyxZQUFZLENBQUM7SUFDM0MsTUFBTSxXQUFXLEdBQUcsUUFBUSxDQUFDLFdBQVcsQ0FBQztJQUN6QyxNQUFNLGNBQWMsR0FBRyxRQUFRLENBQUMsY0FBYyxDQUFDO0lBQy9DLE1BQU0sYUFBYSxHQUFHLFFBQVEsQ0FBQyxhQUFhLENBQUM7SUFDN0MsTUFBTSxxQkFBcUIsR0FBRyxRQUFRLENBQUMscUJBQXFCLENBQUM7SUFDN0QsTUFBTSxvQkFBb0IsR0FBRyxRQUFRLENBQUMsb0JBQW9CLENBQUM7SUFDM0QsTUFBTSxNQUFNLEdBQUcsUUFBUSxDQUFDLE9BQU8sQ0FBQyxHQUFHLENBQUM7SUFDcEMsTUFBTSxPQUFPLEdBQUcsUUFBUSxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUM7SUFFdEMsTUFBTSxZQUFZLEdBQ2QsQ0FBQyxRQUFRLEtBQUssS0FBSyxDQUFDLENBQUMsQ0FBQyxNQUFNLENBQUMsaUJBQWlCLENBQUMsQ0FBQztRQUMxQixNQUFNLENBQUMsaUJBQWlCLENBQUMsQ0FBQztJQUVwRCxNQUFNLE1BQU0sR0FBRyxNQUFNLENBQUMsUUFBUSxDQUFDLFFBQVEsRUFBRSxLQUFLLENBQUMsQ0FBQztJQUNoRCxNQUFNLFVBQVUsR0FBRyxNQUFNLENBQUMsTUFBTSxDQUFDO0lBRWpDLE1BQU0sa0JBQWtC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b0JBQW9CO2dEQUNyRCxJQUFJLEdBQUcscUJBQXFCLEdBQUcsSUFBSSxDQUFDO3FDQUN6QztpQ0FDRjs2QkFDRjt5QkFDRjt3QkFFRCxZQUFZLENBQUMsR0FBRyxDQUFDLFdBQVcsRUFBRSxLQUFLLEVBQUUsTUFBTSxFQUFFLElBQUksRUFBRSxJQUFJLEVBQUUsT0FBTyxDQUFDLENBQUM7cUJBQ25FO2lCQUNGO2FBQ0Y7U0FDRjtLQUNGO0lBRUQsT0FBTyxZQUFZLENBQUM7QUFDdEIsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtiYWNrZW5kX3V0aWwsIGJ1ZmZlciwgRGF0YVR5cGUsIFJhbmssIFRlbnNvckJ1ZmZlciwgVHlwZWRBcnJheX0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuZXhwb3J0IGZ1bmN0aW9uIHBvb2woXG4gICAgeFZhbHVlczogVHlwZWRBcnJheSwgeFNoYXBlOiBudW1iZXJbXSwgZHR5cGU6IERhdGFUeXBlLCBzdHJpZGVzOiBudW1iZXJbXSxcbiAgICBjb252SW5mbzogYmFja2VuZF91dGlsLkNvbnYyREluZm8sXG4gICAgcG9vbFR5cGU6ICdtYXgnfCdhdmcnKTogVGVuc29yQnVmZmVyPFJhbmssIERhdGFUeXBlPiB7XG4gIGNvbnN0IHN0cmlkZUhlaWdodCA9IGNvbnZJbmZvLnN0cmlkZUhlaWdodDtcbiAgY29uc3Qgc3RyaWRlV2lkdGggPSBjb252SW5mby5zdHJpZGVXaWR0aDtcbiAgY29uc3QgZGlsYXRpb25IZWlnaHQgPSBjb252SW5mby5kaWxhdGlvbkhlaWdodDtcbiAgY29uc3QgZGlsYXRpb25XaWR0aCA9IGNvbnZJbmZvLmRpbGF0aW9uV2lkdGg7XG4gIGNvbnN0IGVmZmVjdGl2ZUZpbHRlckhlaWdodCA9IGNvbnZJbmZvLmVmZmVjdGl2ZUZpbHRlckhlaWdodDtcbiAgY29uc3QgZWZmZWN0aXZlRmlsdGVyV2lkdGggPSBjb252SW5mby5lZmZlY3RpdmVGaWx0ZXJXaWR0aDtcbiAgY29uc3QgcGFkVG9wID0gY29udkluZm8ucGFkSW5mby50b3A7XG4gIGNvbnN0IHBhZExlZnQgPSBjb252SW5mby5wYWRJbmZvLmxlZnQ7XG5cbiAgY29uc3QgaW5pdGlhbFZhbHVlID1cbiAgICAgIChwb29sVHlwZSA9PT0gJ21heCcgPyBOdW1iZXIuTkVHQVRJVkVfSU5GSU5JVFkgOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIE51bWJlci5QT1NJVElWRV9JTkZJTklUWSk7XG5cbiAgY29uc3Qgb3V0cHV0ID0gYnVmZmVyKGNvbnZJbmZvLm91dFNoYXBlLCBkdHlwZSk7XG4gIGNvbnN0IG91dHB1dFZhbHMgPSBvdXRwdXQudmFsdWVzO1xuXG4gIGNvbnN0IG91dHB1dEJhdGNoU3RyaWRlcyA9XG4gICAgICBjb252SW5mby5vdXRTaGFwZVsxXSAqIGNvbnZJbmZvLm91dFNoYXBlWzJdICogY29udkluZm8ub3V0U2hhcGVbM107XG4gIGNvbnN0IG91dHB1dFJvd1N0cmlkZXMgPSBjb252SW5mby5vdXRTaGFwZVsyXSAqIGNvbnZJbmZvLm91dFNoYXBlWzNdO1xuICBjb25zdCBvdXRwdXRDb2xTdHJpZGVzID0gY29udkluZm8ub3V0U2hhcGVbM107XG5cbiAgZm9yIChsZXQgYiA9IDA7IGIgPCBjb252SW5mby5iYXRjaFNpemU7ICsrYikge1xuICAgIGNvbnN0IG91dHB1dEJhdGNoT2Zmc2V0ID0gYiAqIG91dHB1dEJhdGNoU3RyaWRlcztcbiAgICBjb25zdCBpbnB1dEJhdGNoT2Zmc2V0ID0gYiAqIHN0cmlkZXNbMF07XG4gICAgZm9yIChsZXQgZCA9IDA7IGQgPCBjb252SW5mby5pbkNoYW5uZWxzOyArK2QpIHtcbiAgICAgIGZvciAobGV0IHlSID0gMDsgeVIgPCBjb252SW5mby5vdXRIZWlnaHQ7ICsreVIpIHtcbiAgICAgICAgY29uc3QgeFJDb3JuZXIgPSB5UiAqIHN0cmlkZUhlaWdodCAtIHBhZFRvcDtcbiAgICAgICAgY29uc3QgeFJNaW4gPSBNYXRoLm1heCgwLCB4UkNvcm5lcik7XG4gICAgICAgIGNvbnN0IHhSTWF4ID1cbiAgICAgICAgICAgIE1hdGgubWluKGNvbnZJbmZvLmluSGVpZ2h0LCBlZmZlY3RpdmVGaWx0ZXJIZWlnaHQgKyB4UkNvcm5lcik7XG4gICAgICAgIGNvbnN0IG91dHB1dFJvd09mZnNldCA9IG91dHB1dEJhdGNoT2Zmc2V0ICsgeVIgKiBvdXRwdXRSb3dTdHJpZGVzO1xuICAgICAgICBmb3IgKGxldCB5QyA9IDA7IHlDIDwgY29udkluZm8ub3V0V2lkdGg7ICsreUMpIHtcbiAgICAgICAgICBjb25zdCB4Q0Nvcm5lciA9IHlDICogc3RyaWRlV2lkdGggLSBwYWRMZWZ0O1xuICAgICAgICAgIGNvbnN0IHhDTWluID0gTWF0aC5tYXgoMCwgeENDb3JuZXIpO1xuICAgICAgICAgIGNvbnN0IHhDTWF4ID1cbiAgICAgICAgICAgICAgTWF0aC5taW4oY29udkluZm8uaW5XaWR0aCwgZWZmZWN0aXZlRmlsdGVyV2lkdGggKyB4Q0Nvcm5lcik7XG4gICAgICAgICAgbGV0IG1pbk1heFZhbHVlID0gaW5pdGlhbFZhbHVlO1xuICAgICAgICAgIGxldCBhdmdWYWx1ZSA9IDA7XG4gICAgICAgICAgbGV0IGNvdW50ID0gMDtcbiAgICAgICAgICBmb3IgKGxldCB4UiA9IHhSTWluOyB4UiA8IHhSTWF4OyB4UiArPSBkaWxhdGlvbkhlaWdodCkge1xuICAgICAgICAgICAgY29uc3QgeFJPZmZzZXQgPSBpbnB1dEJhdGNoT2Zmc2V0ICsgeFIgKiBzdHJpZGVzWzFdO1xuICAgICAgICAgICAgZm9yIChsZXQgeEMgPSB4Q01pbjsgeEMgPCB4Q01heDsgeEMgKz0gZGlsYXRpb25XaWR0aCkge1xuICAgICAgICAgICAgICBjb25zdCB4Q09mZnNldCA9IHhST2Zmc2V0ICsgeEMgKiBzdHJpZGVzWzJdO1xuICAgICAgICAgICAgICBjb25zdCBwaXhlbCA9IHhWYWx1ZXNbeENPZmZzZXQgKyBkXTtcbiAgICAgICAgICAgICAgaWYgKChwb29sVHlwZSA9PT0gJ21heCcgJiYgcGl4ZWwgPiBtaW5NYXhWYWx1ZSkpIHtcbiAgICAgICAgICAgICAgICBtaW5NYXhWYWx1ZSA9IHBpeGVsO1xuICAgICAgICAgICAgICB9IGVsc2UgaWYgKHBvb2xUeXBlID09PSAnYXZnJykge1xuICAgICAgICAgICAgICAgIGF2Z1ZhbHVlICs9IHBpeGVsO1xuICAgICAgICAgICAgICAgIGNvdW50Kys7XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGlmIChpc05hTihtaW5NYXhWYWx1ZSkpIHtcbiAgICAgICAgICAgICAgYnJlYWs7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICAgIGNvbnN0IG91dHB1dE9mZnNldCA9IG91dHB1dFJvd09mZnNldCArIHlDICogb3V0cHV0Q29sU3RyaWRlcyArIGQ7XG4gICAgICAgICAgb3V0cHV0VmFsc1tvdXRwdXRPZmZzZXRdID1cbiAgICAgICAgICAgICAgcG9vbFR5cGUgPT09ICdhdmcnID8gYXZnVmFsdWUgLyBjb3VudCA6IG1pbk1heFZhbHVlO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9XG4gIHJldHVybiBvdXRwdXQ7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBtYXhQb29sUG9zaXRpb25zKFxuICAgIHhWYWx1ZXM6IFR5cGVkQXJyYXksIHhTaGFwZTogbnVtYmVyW10sIGR0eXBlOiBEYXRhVHlwZSxcbiAgICBjb252SW5mbzogYmFja2VuZF91dGlsLkNvbnYyREluZm8sIGZsYXR0ZW5Qb3NpdGlvbnMgPSBmYWxzZSxcbiAgICBpbmNsdWRlQmF0Y2hJbkluZGV4ID0gZmFsc2UpOiBUZW5zb3JCdWZmZXI8UmFuaywgJ2ludDMyJz4ge1xuICBjb25zdCBtYXhQb3NpdGlvbnMgPSBidWZmZXIoY29udkluZm8ub3V0U2hhcGUsICdpbnQzMicpO1xuICBjb25zdCBzdHJpZGVIZWlnaHQgPSBjb252SW5mby5zdHJpZGVIZWlnaHQ7XG4gIGNvbnN0IHN0cmlkZVdpZHRoID0gY29udkluZm8uc3RyaWRlV2lkdGg7XG4gIGNvbnN0IGRpbGF0aW9uSGVpZ2h0ID0gY29udkluZm8uZGlsYXRpb25IZWlnaHQ7XG4gIGNvbnN0IGRpbGF0aW9uV2lkdGggPSBjb252SW5mby5kaWxhdGlvbldpZHRoO1xuICBjb25zdCBlZmZlY3RpdmVGaWx0ZXJIZWlnaHQgPSBjb252SW5mby5lZmZlY3RpdmVGaWx0ZXJIZWlnaHQ7XG4gIGNvbnN0IGVmZmVjdGl2ZUZpbHRlcldpZHRoID0gY29udkluZm8uZWZmZWN0aXZlRmlsdGVyV2lkdGg7XG4gIGNvbnN0IHBhZFRvcCA9IGNvbnZJbmZvLnBhZEluZm8udG9wO1xuICBjb25zdCBwYWRMZWZ0ID0gY29udkluZm8ucGFkSW5mby5sZWZ0O1xuXG4gIGNvbnN0IHhCdWYgPSBidWZmZXIoeFNoYXBlLCBkdHlwZSwgeFZhbHVlcyk7XG4gIGZvciAobGV0IGIgPSAwOyBiIDwgY29udkluZm8uYmF0Y2hTaXplOyArK2IpIHtcbiAgICBmb3IgKGxldCBkID0gMDsgZCA8IGNvbnZJbmZvLmluQ2hhbm5lbHM7ICsrZCkge1xuICAgICAgZm9yIChsZXQgeVIgPSAwOyB5UiA8IGNvbnZJbmZvLm91dEhlaWdodDsgKyt5Uikge1xuICAgICAgICBjb25zdCB4UkNvcm5lciA9IHlSICogc3RyaWRlSGVpZ2h0IC0gcGFkVG9wO1xuICAgICAgICBsZXQgeFJNaW4gPSB4UkNvcm5lcjtcbiAgICAgICAgd2hpbGUgKHhSTWluIDwgMCkge1xuICAgICAgICAgIHhSTWluICs9IGRpbGF0aW9uSGVpZ2h0O1xuICAgICAgICB9XG4gICAgICAgIC8vIGNvbnN0IHhSTWluID0gTWF0aC5tYXgoMCwgeFJDb3JuZXIpO1xuICAgICAgICBjb25zdCB4Uk1heCA9XG4gICAgICAgICAgICBNYXRoLm1pbihjb252SW5mby5pbkhlaWdodCwgZWZmZWN0aXZlRmlsdGVySGVpZ2h0ICsgeFJDb3JuZXIpO1xuICAgICAgICBmb3IgKGxldCB5QyA9IDA7IHlDIDwgY29udkluZm8ub3V0V2lkdGg7ICsreUMpIHtcbiAgICAgICAgICBjb25zdCB4Q0Nvcm5lciA9IHlDICogc3RyaWRlV2lkdGggLSBwYWRMZWZ0O1xuICAgICAgICAgIGxldCB4Q01pbiA9IHhDQ29ybmVyO1xuICAgICAgICAgIHdoaWxlICh4Q01pbiA8IDApIHtcbiAgICAgICAgICAgIHhDTWluICs9IGRpbGF0aW9uV2lkdGg7XG4gICAgICAgICAgfVxuICAgICAgICAgIGNvbnN0IHhDTWF4ID1cbiAgICAgICAgICAgICAgTWF0aC5taW4oY29udkluZm8uaW5XaWR0aCwgZWZmZWN0aXZlRmlsdGVyV2lkdGggKyB4Q0Nvcm5lcik7XG4gICAgICAgICAgbGV0IG1heFZhbHVlID0gTnVtYmVyLk5FR0FUSVZFX0lORklOSVRZO1xuICAgICAgICAgIGxldCBtYXhQb3NpdGlvbiA9IC0xO1xuXG4gICAgICAgICAgZm9yIChsZXQgeFIgPSB4Uk1pbjsgeFIgPCB4Uk1heDsgeFIgKz0gZGlsYXRpb25IZWlnaHQpIHtcbiAgICAgICAgICAgIGNvbnN0IHdSID0geFIgLSB4UkNvcm5lcjtcbiAgICAgICAgICAgIGZvciAobGV0IHhDID0geENNaW47IHhDIDwgeENNYXg7IHhDICs9IGRpbGF0aW9uV2lkdGgpIHtcbiAgICAgICAgICAgICAgY29uc3Qgd0MgPSB4QyAtIHhDQ29ybmVyO1xuICAgICAgICAgICAgICAvLyBGb3Igc29tZSByZWFzb24sIGRpc2FibGUtbmV4dC1saW5lIGlzIG5vdCB3b3JraW5nXG4gICAgICAgICAgICAgIC8vIFRPRE8obWF0dHNvdWxhbmlsbGUpOiBSZW1vdmUgdGhpcyB3aGVuIHN3aXRjaGluZyB0byBUUzUuXG4gICAgICAgICAgICAgIC8qIHRzbGludDpkaXNhYmxlOiBuby11bm5lY2Vzc2FyeS10eXBlLWFzc2VydGlvbiAqL1xuICAgICAgICAgICAgICBjb25zdCBwaXhlbCA9IHhCdWYuZ2V0KGIsIHhSLCB4QywgZCkgYXMgbnVtYmVyO1xuICAgICAgICAgICAgICBpZiAocGl4ZWwgPiBtYXhWYWx1ZSkge1xuICAgICAgICAgICAgICAgIG1heFZhbHVlID0gcGl4ZWwgYXMgbnVtYmVyO1xuICAgICAgICAgICAgICAgIGlmIChmbGF0dGVuUG9zaXRpb25zKSB7XG4gICAgICAgICAgICAgICAgICBtYXhQb3NpdGlvbiA9IGluY2x1ZGVCYXRjaEluSW5kZXggP1xuICAgICAgICAgICAgICAgICAgICAgICgoYiAqIGNvbnZJbmZvLmluSGVpZ2h0ICsgeFIpICogY29udkluZm8uaW5XaWR0aCArIHhDKSAqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICBjb252SW5mby5pbkNoYW5uZWxzICtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgZCA6XG4gICAgICAgICAgICAgICAgICAgICAgKHhSICogY29udkluZm8uaW5XaWR0aCArIHhDKSAqIGNvbnZJbmZvLmluQ2hhbm5lbHMgKyBkO1xuICAgICAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgICBtYXhQb3NpdGlvbiA9IHdSICogZWZmZWN0aXZlRmlsdGVyV2lkdGggKyB3QztcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgICAgbWF4UG9zaXRpb25zLnNldChtYXhQb3NpdGlvbiwgYiwgeVIsIHlDLCBkKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgfVxuICByZXR1cm4gbWF4UG9zaXRpb25zO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gcG9vbDNkKFxuICAgIHhWYWx1ZXM6IFR5cGVkQXJyYXksIHhTaGFwZTogbnVtYmVyW10sIGR0eXBlOiBEYXRhVHlwZSwgc3RyaWRlczogbnVtYmVyW10sXG4gICAgY29udkluZm86IGJhY2tlbmRfdXRpbC5Db252M0RJbmZvLFxuICAgIHBvb2xUeXBlOiAnbWF4J3wnYXZnJyk6IFRlbnNvckJ1ZmZlcjxSYW5rLCBEYXRhVHlwZT4ge1xuICBjb25zdCBzdHJpZGVEZXB0aCA9IGNvbnZJbmZvLnN0cmlkZURlcHRoO1xuICBjb25zdCBzdHJpZGVIZWlnaHQgPSBjb252SW5mby5zdHJpZGVIZWlnaHQ7XG4gIGNvbnN0IHN0cmlkZVdpZHRoID0gY29udkluZm8uc3RyaWRlV2lkdGg7XG4gIGNvbnN0IGRpbGF0aW9uRGVwdGggPSBjb252SW5mby5kaWxhdGlvbkRlcHRoO1xuICBjb25zdCBkaWxhdGlvbkhlaWdodCA9IGNvbnZJbmZvLmRpbGF0aW9uSGVpZ2h0O1xuICBjb25zdCBkaWxhdGlvbldpZHRoID0gY29udkluZm8uZGlsYXRpb25XaWR0aDtcbiAgY29uc3QgZWZmZWN0aXZlRmlsdGVyRGVwdGggPSBjb252SW5mby5lZmZlY3RpdmVGaWx0ZXJEZXB0aDtcbiAgY29uc3QgZWZmZWN0aXZlRmlsdGVySGVpZ2h0ID0gY29udkluZm8uZWZmZWN0aXZlRmlsdGVySGVpZ2h0O1xuICBjb25zdCBlZmZlY3RpdmVGaWx0ZXJXaWR0aCA9IGNvbnZJbmZvLmVmZmVjdGl2ZUZpbHRlcldpZHRoO1xuICBjb25zdCBwYWRGcm9udCA9IGNvbnZJbmZvLnBhZEluZm8uZnJvbnQ7XG4gIGNvbnN0IHBhZFRvcCA9IGNvbnZJbmZvLnBhZEluZm8udG9wO1xuICBjb25zdCBwYWRMZWZ0ID0gY29udkluZm8ucGFkSW5mby5sZWZ0O1xuXG4gIGNvbnN0IGluaXRpYWxWYWx1ZSA9XG4gICAgICAocG9vbFR5cGUgPT09ICdtYXgnID8gTnVtYmVyLk5FR0FUSVZFX0lORklOSVRZIDpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBOdW1iZXIuUE9TSVRJVkVfSU5GSU5JVFkpO1xuXG4gIGNvbnN0IG91dHB1dCA9IGJ1ZmZlcihjb252SW5mby5vdXRTaGFwZSwgZHR5cGUpO1xuICBjb25zdCBvdXRwdXRWYWxzID0gb3V0cHV0LnZhbHVlcztcblxuICBjb25zdCBvdXRwdXRCYXRjaFN0cmlkZXMgPSBjb252SW5mby5vdXRTaGFwZVsxXSAqIGNvbnZJbmZvLm91dFNoYXBlWzJdICpcbiAgICAgIGNvbnZJbmZvLm91dFNoYXBlWzNdICogY29udkluZm8ub3V0U2hhcGVbNF07XG4gIGNvbnN0IG91dHB1dERlcHRoU3RyaWRlcyA9XG4gICAgICBjb252SW5mby5vdXRTaGFwZVsyXSAqIGNvbnZJbmZvLm91dFNoYXBlWzNdICogY29udkluZm8ub3V0U2hhcGVbNF07XG4gIGNvbnN0IG91dHB1dFJvd1N0cmlkZXMgPSBjb252SW5mby5vdXRTaGFwZVszXSAqIGNvbnZJbmZvLm91dFNoYXBlWzRdO1xuICBjb25zdCBvdXRwdXRDb2xTdHJpZGVzID0gY29udkluZm8ub3V0U2hhcGVbNF07XG5cbiAgZm9yIChsZXQgYmF0Y2ggPSAwOyBiYXRjaCA8IGNvbnZJbmZvLmJhdGNoU2l6ZTsgKytiYXRjaCkge1xuICAgIGNvbnN0IG91dHB1dEJhdGNoT2Zmc2V0ID0gYmF0Y2ggKiBvdXRwdXRCYXRjaFN0cmlkZXM7XG4gICAgY29uc3QgaW5wdXRCYXRjaE9mZnNldCA9IGJhdGNoICogc3RyaWRlc1swXTtcbiAgICBmb3IgKGxldCBjaGFubmVsID0gMDsgY2hhbm5lbCA8IGNvbnZJbmZvLmluQ2hhbm5lbHM7ICsrY2hhbm5lbCkge1xuICAgICAgZm9yIChsZXQgeURlcHRoID0gMDsgeURlcHRoIDwgY29udkluZm8ub3V0RGVwdGg7ICsreURlcHRoKSB7XG4gICAgICAgIGNvbnN0IHhEZXB0aENvcm5lciA9IHlEZXB0aCAqIHN0cmlkZURlcHRoIC0gcGFkRnJvbnQ7XG4gICAgICAgIGxldCB4RGVwdGhNaW4gPSB4RGVwdGhDb3JuZXI7XG4gICAgICAgIHdoaWxlICh4RGVwdGhNaW4gPCAwKSB7XG4gICAgICAgICAgeERlcHRoTWluICs9IGRpbGF0aW9uRGVwdGg7XG4gICAgICAgIH1cbiAgICAgICAgY29uc3QgeERlcHRoTWF4ID1cbiAgICAgICAgICAgIE1hdGgubWluKGNvbnZJbmZvLmluRGVwdGgsIGVmZmVjdGl2ZUZpbHRlckRlcHRoICsgeERlcHRoQ29ybmVyKTtcbiAgICAgICAgY29uc3Qgb3V0cHV0RGVwdGhPZmZzZXQgPVxuICAgICAgICAgICAgb3V0cHV0QmF0Y2hPZmZzZXQgKyB5RGVwdGggKiBvdXRwdXREZXB0aFN0cmlkZXM7XG4gICAgICAgIGZvciAobGV0IHlSb3cgPSAwOyB5Um93IDwgY29udkluZm8ub3V0SGVpZ2h0OyArK3lSb3cpIHtcbiAgICAgICAgICBjb25zdCB4Um93Q29ybmVyID0geVJvdyAqIHN0cmlkZUhlaWdodCAtIHBhZFRvcDtcbiAgICAgICAgICBsZXQgeFJvd01pbiA9IHhSb3dDb3JuZXI7XG4gICAgICAgICAgd2hpbGUgKHhSb3dNaW4gPCAwKSB7XG4gICAgICAgICAgICB4Um93TWluICs9IGRpbGF0aW9uSGVpZ2h0O1xuICAgICAgICAgIH1cbiAgICAgICAgICBjb25zdCB4Um93TWF4ID1cbiAgICAgICAgICAgICAgTWF0aC5taW4oY29udkluZm8uaW5IZWlnaHQsIGVmZmVjdGl2ZUZpbHRlckhlaWdodCArIHhSb3dDb3JuZXIpO1xuICAgICAgICAgIGNvbnN0IG91dHB1dFJvd09mZnNldCA9IG91dHB1dERlcHRoT2Zmc2V0ICsgeVJvdyAqIG91dHB1dFJvd1N0cmlkZXM7XG4gICAgICAgICAgZm9yIChsZXQgeUNvbCA9IDA7IHlDb2wgPCBjb252SW5mby5vdXRXaWR0aDsgKyt5Q29sKSB7XG4gICAgICAgICAgICBjb25zdCB4Q29sQ29ybmVyID0geUNvbCAqIHN0cmlkZVdpZHRoIC0gcGFkTGVmdDtcbiAgICAgICAgICAgIGxldCB4Q29sTWluID0geENvbENvcm5lcjtcbiAgICAgICAgICAgIHdoaWxlICh4Q29sTWluIDwgMCkge1xuICAgICAgICAgICAgICB4Q29sTWluICs9IGRpbGF0aW9uV2lkdGg7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBjb25zdCB4Q29sTWF4ID1cbiAgICAgICAgICAgICAgICBNYXRoLm1pbihjb252SW5mby5pbldpZHRoLCBlZmZlY3RpdmVGaWx0ZXJXaWR0aCArIHhDb2xDb3JuZXIpO1xuICAgICAgICAgICAgLy8gU2hhZGVyIGNvZGUgYmVnaW5zXG4gICAgICAgICAgICBjb25zdCBvdXRwdXRDb2xPZmZzZXQgPSBvdXRwdXRSb3dPZmZzZXQgKyB5Q29sICogb3V0cHV0Q29sU3RyaWRlcztcbiAgICAgICAgICAgIGxldCBtaW5NYXhWYWx1ZSA9IGluaXRpYWxWYWx1ZTtcbiAgICAgICAgICAgIGxldCBhdmdWYWx1ZSA9IDA7XG4gICAgICAgICAgICBsZXQgY291bnQgPSAwO1xuICAgICAgICAgICAgZm9yIChsZXQgeERlcHRoID0geERlcHRoTWluOyB4RGVwdGggPCB4RGVwdGhNYXg7XG4gICAgICAgICAgICAgICAgIHhEZXB0aCArPSBkaWxhdGlvbkRlcHRoKSB7XG4gICAgICAgICAgICAgIGNvbnN0IHhEZXB0aE9mZnNldCA9IGlucHV0QmF0Y2hPZmZzZXQgKyB4RGVwdGggKiBzdHJpZGVzWzFdO1xuICAgICAgICAgICAgICBmb3IgKGxldCB4Um93ID0geFJvd01pbjsgeFJvdyA8IHhSb3dNYXg7IHhSb3cgKz0gZGlsYXRpb25IZWlnaHQpIHtcbiAgICAgICAgICAgICAgICBjb25zdCB4Um93T2Zmc2V0ID0geERlcHRoT2Zmc2V0ICsgeFJvdyAqIHN0cmlkZXNbMl07XG4gICAgICAgICAgICAgICAgZm9yIChsZXQgeENvbCA9IHhDb2xNaW47IHhDb2wgPCB4Q29sTWF4O1xuICAgICAgICAgICAgICAgICAgICAgeENvbCArPSBkaWxhdGlvbldpZHRoKSB7XG4gICAgICAgICAgICAgICAgICBjb25zdCB4Q29sT2Zmc2V0ID0geFJvd09mZnNldCArIHhDb2wgKiBzdHJpZGVzWzNdO1xuICAgICAgICAgICAgICAgICAgY29uc3QgcGl4ZWwgPSB4VmFsdWVzW3hDb2xPZmZzZXQgKyBjaGFubmVsXTtcbiAgICAgICAgICAgICAgICAgIGlmICgocG9vbFR5cGUgPT09ICdtYXgnICYmIHBpeGVsID4gbWluTWF4VmFsdWUpKSB7XG4gICAgICAgICAgICAgICAgICAgIG1pbk1heFZhbHVlID0gcGl4ZWw7XG4gICAgICAgICAgICAgICAgICB9IGVsc2UgaWYgKHBvb2xUeXBlID09PSAnYXZnJykge1xuICAgICAgICAgICAgICAgICAgICBhdmdWYWx1ZSArPSBwaXhlbDtcbiAgICAgICAgICAgICAgICAgICAgY291bnQrKztcbiAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgIGlmIChpc05hTihtaW5NYXhWYWx1ZSkpIHtcbiAgICAgICAgICAgICAgICAgICAgYnJlYWs7XG4gICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIGlmIChpc05hTihtaW5NYXhWYWx1ZSkpIHtcbiAgICAgICAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICBpZiAoaXNOYU4obWluTWF4VmFsdWUpKSB7XG4gICAgICAgICAgICAgICAgYnJlYWs7XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGNvbnN0IG91dHB1dE9mZnNldCA9IG91dHB1dENvbE9mZnNldCArIGNoYW5uZWw7XG4gICAgICAgICAgICBvdXRwdXRWYWxzW291dHB1dE9mZnNldF0gPSBwb29sVHlwZSA9PT0gJ2F2ZycgP1xuICAgICAgICAgICAgICAgIGF2Z1ZhbHVlIC8gTWF0aC5tYXgoY291bnQsIDEpIDpcbiAgICAgICAgICAgICAgICBtaW5NYXhWYWx1ZTtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gIH1cblxuICByZXR1cm4gb3V0cHV0O1xufVxuXG5leHBvcnQgZnVuY3Rpb24gbWF4UG9vbDNkUG9zaXRpb25zKFxuICAgIHhCdWY6IFRlbnNvckJ1ZmZlcjxSYW5rLCBEYXRhVHlwZT4sXG4gICAgY29udkluZm86IGJhY2tlbmRfdXRpbC5Db252M0RJbmZvKTogVGVuc29yQnVmZmVyPFJhbmssIERhdGFUeXBlPiB7XG4gIGNvbnN0IG1heFBvc2l0aW9ucyA9IGJ1ZmZlcihjb252SW5mby5vdXRTaGFwZSwgJ2ludDMyJyk7XG4gIGNvbnN0IHN0cmlkZURlcHRoID0gY29udkluZm8uc3RyaWRlRGVwdGg7XG4gIGNvbnN0IHN0cmlkZUhlaWdodCA9IGNvbnZJbmZvLnN0cmlkZUhlaWdodDtcbiAgY29uc3Qgc3RyaWRlV2lkdGggPSBjb252SW5mby5zdHJpZGVXaWR0aDtcbiAgY29uc3QgZGlsYXRpb25EZXB0aCA9IGNvbnZJbmZvLmRpbGF0aW9uRGVwdGg7XG4gIGNvbnN0IGRpbGF0aW9uSGVpZ2h0ID0gY29udkluZm8uZGlsYXRpb25IZWlnaHQ7XG4gIGNvbnN0IGRpbGF0aW9uV2lkdGggPSBjb252SW5mby5kaWxhdGlvbldpZHRoO1xuICBjb25zdCBlZmZlY3RpdmVGaWx0ZXJEZXB0aCA9IGNvbnZJbmZvLmVmZmVjdGl2ZUZpbHRlckRlcHRoO1xuICBjb25zdCBlZmZlY3RpdmVGaWx0ZXJIZWlnaHQgPSBjb252SW5mby5lZmZlY3RpdmVGaWx0ZXJIZWlnaHQ7XG4gIGNvbnN0IGVmZmVjdGl2ZUZpbHRlcldpZHRoID0gY29udkluZm8uZWZmZWN0aXZlRmlsdGVyV2lkdGg7XG4gIGNvbnN0IHBhZEZyb250ID0gY29udkluZm8ucGFkSW5mby5mcm9udDtcbiAgY29uc3QgcGFkVG9wID0gY29udkluZm8ucGFkSW5mby50b3A7XG4gIGNvbnN0IHBhZExlZnQgPSBjb252SW5mby5wYWRJbmZvLmxlZnQ7XG5cbiAgZm9yIChsZXQgYmF0Y2ggPSAwOyBiYXRjaCA8IGNvbnZJbmZvLmJhdGNoU2l6ZTsgKytiYXRjaCkge1xuICAgIGZvciAobGV0IGNoYW5uZWwgPSAwOyBjaGFubmVsIDwgY29udkluZm8uaW5DaGFubmVsczsgKytjaGFubmVsKSB7XG4gICAgICBmb3IgKGxldCB5RGVwdGggPSAwOyB5RGVwdGggPCBjb252SW5mby5vdXREZXB0aDsgKyt5RGVwdGgpIHtcbiAgICAgICAgY29uc3QgeERlcHRoQ29ybmVyID0geURlcHRoICogc3RyaWRlRGVwdGggLSBwYWRGcm9udDtcbiAgICAgICAgbGV0IHhEZXB0aE1pbiA9IHhEZXB0aENvcm5lcjtcbiAgICAgICAgd2hpbGUgKHhEZXB0aE1pbiA8IDApIHtcbiAgICAgICAgICB4RGVwdGhNaW4gKz0gZGlsYXRpb25EZXB0aDtcbiAgICAgICAgfVxuICAgICAgICBjb25zdCB4RGVwdGhNYXggPVxuICAgICAgICAgICAgTWF0aC5taW4oY29udkluZm8uaW5EZXB0aCwgZWZmZWN0aXZlRmlsdGVyRGVwdGggKyB4RGVwdGhDb3JuZXIpO1xuICAgICAgICBmb3IgKGxldCB5Um93ID0gMDsgeVJvdyA8IGNvbnZJbmZvLm91dEhlaWdodDsgKyt5Um93KSB7XG4gICAgICAgICAgY29uc3QgeFJvd0Nvcm5lciA9IHlSb3cgKiBzdHJpZGVIZWlnaHQgLSBwYWRUb3A7XG4gICAgICAgICAgbGV0IHhSb3dNaW4gPSB4Um93Q29ybmVyO1xuICAgICAgICAgIHdoaWxlICh4Um93TWluIDwgMCkge1xuICAgICAgICAgICAgeFJvd01pbiArPSBkaWxhdGlvbkhlaWdodDtcbiAgICAgICAgICB9XG4gICAgICAgICAgY29uc3QgeFJvd01heCA9XG4gICAgICAgICAgICAgIE1hdGgubWluKGNvbnZJbmZvLmluSGVpZ2h0LCBlZmZlY3RpdmVGaWx0ZXJIZWlnaHQgKyB4Um93Q29ybmVyKTtcbiAgICAgICAgICBmb3IgKGxldCB5Q29sID0gMDsgeUNvbCA8IGNvbnZJbmZvLm91dFdpZHRoOyArK3lDb2wpIHtcbiAgICAgICAgICAgIGNvbnN0IHhDb2xDb3JuZXIgPSB5Q29sICogc3RyaWRlV2lkdGggLSBwYWRMZWZ0O1xuICAgICAgICAgICAgbGV0IHhDb2xNaW4gPSB4Q29sQ29ybmVyO1xuICAgICAgICAgICAgd2hpbGUgKHhDb2xNaW4gPCAwKSB7XG4gICAgICAgICAgICAgIHhDb2xNaW4gKz0gZGlsYXRpb25XaWR0aDtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGNvbnN0IHhDb2xNYXggPVxuICAgICAgICAgICAgICAgIE1hdGgubWluKGNvbnZJbmZvLmluV2lkdGgsIGVmZmVjdGl2ZUZpbHRlcldpZHRoICsgeENvbENvcm5lcik7XG5cbiAgICAgICAgICAgIC8vIFNoYWRlciBjb2RlIGJlZ2luc1xuICAgICAgICAgICAgbGV0IG1heFZhbHVlID0gTnVtYmVyLk5FR0FUSVZFX0lORklOSVRZO1xuICAgICAgICAgICAgbGV0IG1heFBvc2l0aW9uID0gLTE7XG5cbiAgICAgICAgICAgIGZvciAobGV0IHhEZXB0aCA9IHhEZXB0aE1pbjsgeERlcHRoIDwgeERlcHRoTWF4O1xuICAgICAgICAgICAgICAgICB4RGVwdGggKz0gZGlsYXRpb25EZXB0aCkge1xuICAgICAgICAgICAgICBjb25zdCB3RGVwdGggPSB4RGVwdGggLSB4RGVwdGhDb3JuZXI7XG4gICAgICAgICAgICAgIGZvciAobGV0IHhSb3cgPSB4Um93TWluOyB4Um93IDwgeFJvd01heDsgeFJvdyArPSBkaWxhdGlvbkhlaWdodCkge1xuICAgICAgICAgICAgICAgIGNvbnN0IHdSb3cgPSB4Um93IC0geFJvd0Nvcm5lcjtcbiAgICAgICAgICAgICAgICBmb3IgKGxldCB4Q29sID0geENvbE1pbjsgeENvbCA8IHhDb2xNYXg7XG4gICAgICAgICAgICAgICAgICAgICB4Q29sICs9IGRpbGF0aW9uV2lkdGgpIHtcbiAgICAgICAgICAgICAgICAgIGNvbnN0IHdDb2wgPSB4Q29sIC0geENvbENvcm5lcjtcbiAgICAgICAgICAgICAgICAgIGNvbnN0IHBpeGVsID0geEJ1Zi5nZXQoYmF0Y2gsIHhEZXB0aCwgeFJvdywgeENvbCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY2hhbm5lbCkgYXMgbnVtYmVyO1xuICAgICAgICAgICAgICAgICAgaWYgKHBpeGVsID49IG1heFZhbHVlKSB7XG4gICAgICAgICAgICAgICAgICAgIG1heFZhbHVlID0gcGl4ZWwgYXMgbnVtYmVyO1xuICAgICAgICAgICAgICAgICAgICBtYXhQb3NpdGlvbiA9XG4gICAgICAgICAgICAgICAgICAgICAgICB3RGVwdGggKiBlZmZlY3RpdmVGaWx0ZXJIZWlnaHQgKiBlZmZlY3RpdmVGaWx0ZXJXaWR0aCArXG4gICAgICAgICAgICAgICAgICAgICAgICB3Um93ICogZWZmZWN0aXZlRmlsdGVySGVpZ2h0ICsgd0NvbDtcbiAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgbWF4UG9zaXRpb25zLnNldChtYXhQb3NpdGlvbiwgYmF0Y2gsIHlEZXB0aCwgeVJvdywgeUNvbCwgY2hhbm5lbCk7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgcmV0dXJuIG1heFBvc2l0aW9ucztcbn1cbiJdfQ==