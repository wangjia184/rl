/**
 * @license
 * Copyright 2023 Google LLC.
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
import { backend_util, StaticRegexReplace } from '@tensorflow/tfjs-core';
import { staticRegexReplaceImplCPU } from '../kernel_utils/shared';
export function staticRegexReplace(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    if (x.dtype !== 'string') {
        throw new Error('Input must be of datatype string');
    }
    const $x = backend.readSync(x.dataId);
    const stringInput = backend_util.fromUint8ToStringArray($x);
    const output = staticRegexReplaceImplCPU(stringInput, 'string', attrs);
    return backend.makeTensorInfo(x.shape, 'string', output);
}
export const staticRegexReplaceConfig = {
    kernelName: StaticRegexReplace,
    backendName: 'webgl',
    kernelFunc: staticRegexReplace,
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiU3RhdGljUmVnZXhSZXBsYWNlLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdsL3NyYy9rZXJuZWxzL1N0YXRpY1JlZ2V4UmVwbGFjZS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsWUFBWSxFQUEwQyxrQkFBa0IsRUFBZ0UsTUFBTSx1QkFBdUIsQ0FBQztBQUU5SyxPQUFPLEVBQUMseUJBQXlCLEVBQUMsTUFBTSx3QkFBd0IsQ0FBQztBQUVqRSxNQUFNLFVBQVUsa0JBQWtCLENBQUMsSUFJbEM7SUFDQyxNQUFNLEVBQUMsTUFBTSxFQUFFLE9BQU8sRUFBRSxLQUFLLEVBQUMsR0FBRyxJQUFJLENBQUM7SUFDdEMsTUFBTSxFQUFDLENBQUMsRUFBQyxHQUFHLE1BQU0sQ0FBQztJQUVuQixJQUFJLENBQUMsQ0FBQyxLQUFLLEtBQUssUUFBUSxFQUFFO1FBQ3hCLE1BQU0sSUFBSSxLQUFLLENBQUMsa0NBQWtDLENBQUMsQ0FBQztLQUNyRDtJQUVELE1BQU0sRUFBRSxHQUFHLE9BQU8sQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLE1BQU0sQ0FBaUIsQ0FBQztJQUV0RCxNQUFNLFdBQVcsR0FBRyxZQUFZLENBQUMsc0JBQXNCLENBQUMsRUFBRSxDQUFDLENBQUM7SUFDNUQsTUFBTSxNQUFNLEdBQUcseUJBQXlCLENBQUMsV0FBVyxFQUFFLFFBQVEsRUFDckIsS0FBZ0MsQ0FBQyxDQUFDO0lBRTNFLE9BQU8sT0FBTyxDQUFDLGNBQWMsQ0FBQyxDQUFDLENBQUMsS0FBSyxFQUFFLFFBQVEsRUFBRSxNQUFNLENBQUMsQ0FBQztBQUMzRCxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sd0JBQXdCLEdBQWlCO0lBQ3BELFVBQVUsRUFBRSxrQkFBa0I7SUFDOUIsV0FBVyxFQUFFLE9BQU87SUFDcEIsVUFBVSxFQUFFLGtCQUEyQztDQUN4RCxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjMgR29vZ2xlIExMQy5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2JhY2tlbmRfdXRpbCwgS2VybmVsQ29uZmlnLCBLZXJuZWxGdW5jLCBOYW1lZEF0dHJNYXAsIFN0YXRpY1JlZ2V4UmVwbGFjZSwgU3RhdGljUmVnZXhSZXBsYWNlQXR0cnMsIFN0YXRpY1JlZ2V4UmVwbGFjZUlucHV0cywgVGVuc29ySW5mb30gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcbmltcG9ydCB7TWF0aEJhY2tlbmRXZWJHTH0gZnJvbSAnLi4vYmFja2VuZF93ZWJnbCc7XG5pbXBvcnQge3N0YXRpY1JlZ2V4UmVwbGFjZUltcGxDUFV9IGZyb20gJy4uL2tlcm5lbF91dGlscy9zaGFyZWQnO1xuXG5leHBvcnQgZnVuY3Rpb24gc3RhdGljUmVnZXhSZXBsYWNlKGFyZ3M6IHtcbiAgaW5wdXRzOiBTdGF0aWNSZWdleFJlcGxhY2VJbnB1dHMsXG4gIGJhY2tlbmQ6IE1hdGhCYWNrZW5kV2ViR0wsXG4gIGF0dHJzOiBTdGF0aWNSZWdleFJlcGxhY2VBdHRycyxcbn0pOiBUZW5zb3JJbmZvIHtcbiAgY29uc3Qge2lucHV0cywgYmFja2VuZCwgYXR0cnN9ID0gYXJncztcbiAgY29uc3Qge3h9ID0gaW5wdXRzO1xuXG4gIGlmICh4LmR0eXBlICE9PSAnc3RyaW5nJykge1xuICAgIHRocm93IG5ldyBFcnJvcignSW5wdXQgbXVzdCBiZSBvZiBkYXRhdHlwZSBzdHJpbmcnKTtcbiAgfVxuXG4gIGNvbnN0ICR4ID0gYmFja2VuZC5yZWFkU3luYyh4LmRhdGFJZCkgYXMgVWludDhBcnJheVtdO1xuXG4gIGNvbnN0IHN0cmluZ0lucHV0ID0gYmFja2VuZF91dGlsLmZyb21VaW50OFRvU3RyaW5nQXJyYXkoJHgpO1xuICBjb25zdCBvdXRwdXQgPSBzdGF0aWNSZWdleFJlcGxhY2VJbXBsQ1BVKHN0cmluZ0lucHV0LCAnc3RyaW5nJyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBhdHRycyBhcyB1bmtub3duIGFzIE5hbWVkQXR0ck1hcCk7XG5cbiAgcmV0dXJuIGJhY2tlbmQubWFrZVRlbnNvckluZm8oeC5zaGFwZSwgJ3N0cmluZycsIG91dHB1dCk7XG59XG5cbmV4cG9ydCBjb25zdCBzdGF0aWNSZWdleFJlcGxhY2VDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogU3RhdGljUmVnZXhSZXBsYWNlLFxuICBiYWNrZW5kTmFtZTogJ3dlYmdsJyxcbiAga2VybmVsRnVuYzogc3RhdGljUmVnZXhSZXBsYWNlIGFzIHVua25vd24gYXMgS2VybmVsRnVuYyxcbn07XG4iXX0=