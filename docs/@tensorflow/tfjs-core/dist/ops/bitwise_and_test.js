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
import * as tf from '../index';
import { ALL_ENVS, describeWithFlags } from '../jasmine_util';
import { expectArraysClose } from '../test_util';
import { bitwiseAnd } from './bitwise_and';
describeWithFlags('bitwiseAnd', ALL_ENVS, () => {
    it('a bitwiseAnd b', async () => {
        const a = tf.tensor1d([0, 5, 3, 14], 'int32');
        const b = tf.tensor1d([5, 0, 7, 11], 'int32');
        const res = bitwiseAnd(a, b);
        expectArraysClose(await res.data(), [0, 0, 3, 10]);
    });
    it('different shape', () => {
        const a = tf.tensor1d([0, 5, 3, 14]);
        const b = tf.tensor1d([5, 0, 7]);
        expect(() => bitwiseAnd(a, b))
            .toThrowError(/BitwiseAnd: Tensors must have the same shape/);
    });
    it('wrong type', () => {
        const a = tf.tensor1d([0, 1, 3, 14], 'float32');
        const b = tf.tensor1d([5, 0, 7, 12], 'float32');
        expect(() => bitwiseAnd(a, b))
            .toThrowError(/BitwiseAnd: Only supports 'int32' values in tensor/);
    });
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYml0d2lzZV9hbmRfdGVzdC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL2JpdHdpc2VfYW5kX3Rlc3QudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBQ0gsT0FBTyxLQUFLLEVBQUUsTUFBTSxVQUFVLENBQUM7QUFDL0IsT0FBTyxFQUFDLFFBQVEsRUFBRSxpQkFBaUIsRUFBQyxNQUFNLGlCQUFpQixDQUFDO0FBQzVELE9BQU8sRUFBQyxpQkFBaUIsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUMvQyxPQUFPLEVBQUMsVUFBVSxFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRXpDLGlCQUFpQixDQUFDLFlBQVksRUFBRSxRQUFRLEVBQUUsR0FBRyxFQUFFO0lBQzdDLEVBQUUsQ0FBQyxnQkFBZ0IsRUFBRSxLQUFLLElBQUksRUFBRTtRQUM5QixNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsT0FBTyxDQUFDLENBQUM7UUFDOUMsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxDQUFDO1FBRTlDLE1BQU0sR0FBRyxHQUFHLFVBQVUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDN0IsaUJBQWlCLENBQUMsTUFBTSxHQUFHLENBQUMsSUFBSSxFQUFFLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQ3JELENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLGlCQUFpQixFQUFFLEdBQUcsRUFBRTtRQUN6QixNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUNyQyxNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBRWpDLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxVQUFVLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO2FBQ3pCLFlBQVksQ0FBQyw4Q0FBOEMsQ0FBQyxDQUFDO0lBQ3BFLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLFlBQVksRUFBRSxHQUFHLEVBQUU7UUFDcEIsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLFNBQVMsQ0FBQyxDQUFDO1FBQ2hELE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRSxTQUFTLENBQUMsQ0FBQztRQUVoRCxNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsVUFBVSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQzthQUN6QixZQUFZLENBQUMsb0RBQW9ELENBQUMsQ0FBQztJQUMxRSxDQUFDLENBQUMsQ0FBQztBQUNMLENBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjMgR29vZ2xlIExMQy5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuaW1wb3J0ICogYXMgdGYgZnJvbSAnLi4vaW5kZXgnO1xuaW1wb3J0IHtBTExfRU5WUywgZGVzY3JpYmVXaXRoRmxhZ3N9IGZyb20gJy4uL2phc21pbmVfdXRpbCc7XG5pbXBvcnQge2V4cGVjdEFycmF5c0Nsb3NlfSBmcm9tICcuLi90ZXN0X3V0aWwnO1xuaW1wb3J0IHtiaXR3aXNlQW5kfSBmcm9tICcuL2JpdHdpc2VfYW5kJztcblxuZGVzY3JpYmVXaXRoRmxhZ3MoJ2JpdHdpc2VBbmQnLCBBTExfRU5WUywgKCkgPT4ge1xuICBpdCgnYSBiaXR3aXNlQW5kIGInLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgYSA9IHRmLnRlbnNvcjFkKFswLCA1LCAzLCAxNF0sICdpbnQzMicpO1xuICAgIGNvbnN0IGIgPSB0Zi50ZW5zb3IxZChbNSwgMCwgNywgMTFdLCAnaW50MzInKTtcblxuICAgIGNvbnN0IHJlcyA9IGJpdHdpc2VBbmQoYSwgYik7XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgcmVzLmRhdGEoKSwgWzAsIDAsIDMsIDEwXSk7XG4gIH0pO1xuXG4gIGl0KCdkaWZmZXJlbnQgc2hhcGUnLCAoKSA9PiB7XG4gICAgY29uc3QgYSA9IHRmLnRlbnNvcjFkKFswLCA1LCAzLCAxNF0pO1xuICAgIGNvbnN0IGIgPSB0Zi50ZW5zb3IxZChbNSwgMCwgN10pO1xuXG4gICAgZXhwZWN0KCgpID0+IGJpdHdpc2VBbmQoYSwgYikpXG4gICAgICAgIC50b1Rocm93RXJyb3IoL0JpdHdpc2VBbmQ6IFRlbnNvcnMgbXVzdCBoYXZlIHRoZSBzYW1lIHNoYXBlLyk7XG4gIH0pO1xuXG4gIGl0KCd3cm9uZyB0eXBlJywgKCkgPT4ge1xuICAgIGNvbnN0IGEgPSB0Zi50ZW5zb3IxZChbMCwgMSwgMywgMTRdLCAnZmxvYXQzMicpO1xuICAgIGNvbnN0IGIgPSB0Zi50ZW5zb3IxZChbNSwgMCwgNywgMTJdLCAnZmxvYXQzMicpO1xuXG4gICAgZXhwZWN0KCgpID0+IGJpdHdpc2VBbmQoYSwgYikpXG4gICAgICAgIC50b1Rocm93RXJyb3IoL0JpdHdpc2VBbmQ6IE9ubHkgc3VwcG9ydHMgJ2ludDMyJyB2YWx1ZXMgaW4gdGVuc29yLyk7XG4gIH0pO1xufSk7XG4iXX0=