/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
import { expectArraysEqual } from '../test_util';
import { CompositeArrayBuffer } from './composite_array_buffer';
describe('CompositeArrayBuffer', () => {
    const uniformBuffers = [
        new Uint8Array([0, 1, 2, 3]).buffer,
        new Uint8Array([4, 5, 6, 7]).buffer,
        new Uint8Array([8, 9, 10, 11]).buffer,
        new Uint8Array([12, 13, 14, 15]).buffer,
        new Uint8Array([16]).buffer,
    ];
    const nonUniformBuffers = [
        new Uint8Array([0, 1, 2]).buffer,
        new Uint8Array([3, 4, 5, 6, 7]).buffer,
        new Uint8Array([8, 9, 10, 11]).buffer,
        new Uint8Array([12, 13, 14, 15, 16]).buffer,
    ];
    const bufferTestCases = [
        ['uniform', uniformBuffers],
        ['non-uniform', nonUniformBuffers]
    ];
    for (const [buffersType, buffers] of bufferTestCases) {
        let composite;
        beforeEach(() => {
            composite = new CompositeArrayBuffer(buffers);
        });
        it(`${buffersType}: slices across multiple buffers`, () => {
            expectArraysEqual(new Uint8Array(composite.slice(1, 13)), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        });
        it(`${buffersType}: slices to the end of the array when \'end\' is not ` +
            'specified', () => {
            expectArraysEqual(new Uint8Array(composite.slice(5)), [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        });
        it(`${buffersType}: makes a copy when slice() is called with no arguments`, () => {
            expectArraysEqual(new Uint8Array(composite.slice()), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        });
        it(`${buffersType}: slices from zero when start is negative`, () => {
            expectArraysEqual(new Uint8Array(composite.slice(-4, 5)), [0, 1, 2, 3, 4]);
        });
        it(`${buffersType}: slices to the end when end is greater than length`, () => {
            expectArraysEqual(new Uint8Array(composite.slice(7, 1000)), [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        });
        it(`${buffersType}: slices multiple ranges out of order`, () => {
            expectArraysEqual(new Uint8Array(composite.slice(13, 15)), [13, 14]);
            expectArraysEqual(new Uint8Array(composite.slice(0, 2)), [0, 1]);
            expectArraysEqual(new Uint8Array(composite.slice(9, 13)), [9, 10, 11, 12]);
        });
    }
    it('can be created from an empty arraybuffer', () => {
        const array = new Uint8Array([]);
        const singleComposite = new CompositeArrayBuffer(array.buffer);
        expectArraysEqual(new Uint8Array(singleComposite.slice()), []);
    });
    it('can be created from a single array', () => {
        const array = new Uint8Array([1, 2, 3]);
        const singleComposite = new CompositeArrayBuffer(array.buffer);
        expectArraysEqual(new Uint8Array(singleComposite.slice()), array);
    });
    it('can be created from zero arrays', () => {
        const singleComposite = new CompositeArrayBuffer([]);
        expectArraysEqual(new Uint8Array(singleComposite.slice()), new Uint8Array());
    });
    it('can be created from undefined input', () => {
        const singleComposite = new CompositeArrayBuffer();
        expectArraysEqual(new Uint8Array(singleComposite.slice()), new Uint8Array());
    });
    it('treats NaN as zero when passed as the start of slice', () => {
        const array = new Uint8Array([1, 2, 3]);
        const composite = new CompositeArrayBuffer(array.buffer);
        expectArraysEqual(new Uint8Array(composite.slice(NaN, 2)), [1, 2]);
    });
    it('treats NaN as zero when passed as the end of slice', () => {
        const array = new Uint8Array([1, 2, 3]);
        const composite = new CompositeArrayBuffer(array.buffer);
        expectArraysEqual(new Uint8Array(composite.slice(0, NaN)), []);
    });
    it('supports TypedArray input', () => {
        // This support is necessary for some tests in tfjs-converter. Maybe those
        // tests are misconfigured?
        const array = new Uint8Array([1, 2, 3]);
        const composite = new CompositeArrayBuffer(array);
        expectArraysEqual(new Uint8Array(composite.slice(0, 2)), [1, 2]);
    });
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29tcG9zaXRlX2FycmF5X2J1ZmZlcl90ZXN0LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9pby9jb21wb3NpdGVfYXJyYXlfYnVmZmVyX3Rlc3QudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBQ0gsT0FBTyxFQUFDLGlCQUFpQixFQUFDLE1BQU0sY0FBYyxDQUFDO0FBQy9DLE9BQU8sRUFBQyxvQkFBb0IsRUFBQyxNQUFNLDBCQUEwQixDQUFDO0FBRTlELFFBQVEsQ0FBQyxzQkFBc0IsRUFBRSxHQUFHLEVBQUU7SUFDcEMsTUFBTSxjQUFjLEdBQUc7UUFDckIsSUFBSSxVQUFVLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLE1BQU07UUFDbkMsSUFBSSxVQUFVLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLE1BQU07UUFDbkMsSUFBSSxVQUFVLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUUsRUFBRSxFQUFFLENBQUMsQ0FBQyxDQUFDLE1BQU07UUFDckMsSUFBSSxVQUFVLENBQUMsQ0FBQyxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLENBQUMsQ0FBQyxDQUFDLE1BQU07UUFDdkMsSUFBSSxVQUFVLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLE1BQU07S0FDNUIsQ0FBQztJQUVGLE1BQU0saUJBQWlCLEdBQUc7UUFDeEIsSUFBSSxVQUFVLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsTUFBTTtRQUNoQyxJQUFJLFVBQVUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FB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QUMsQ0FBQyxDQUFDLEVBQUMsQ0FBQyxFQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDdEMsTUFBTSxTQUFTLEdBQUcsSUFBSSxvQkFBb0IsQ0FBQyxLQUFLLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDekQsaUJBQWlCLENBQUMsSUFBSSxVQUFVLENBQUMsU0FBUyxDQUFDLEtBQUssQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLENBQUMsRUFBRSxFQUFFLENBQUMsQ0FBQztJQUNqRSxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQywyQkFBMkIsRUFBRSxHQUFHLEVBQUU7UUFDbkMsMEVBQTBFO1FBQzFFLDJCQUEyQjtRQUMzQixNQUFNLEtBQUssR0FBRyxJQUFJLFVBQVUsQ0FBQyxDQUFDLENBQUMsRUFBQyxDQUFDLEVBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUN0QyxNQUFNLFNBQVMsR0FBRyxJQUFJLG9CQUFvQixDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQ2xELGlCQUFpQixDQUFDLElBQUksVUFBVSxDQUFDLFNBQVMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUNsRSxDQUFDLENBQUMsQ0FBQztBQUNMLENBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjMgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuaW1wb3J0IHtleHBlY3RBcnJheXNFcXVhbH0gZnJvbSAnLi4vdGVzdF91dGlsJztcbmltcG9ydCB7Q29tcG9zaXRlQXJyYXlCdWZmZXJ9IGZyb20gJy4vY29tcG9zaXRlX2FycmF5X2J1ZmZlcic7XG5cbmRlc2NyaWJlKCdDb21wb3NpdGVBcnJheUJ1ZmZlcicsICgpID0+IHtcbiAgY29uc3QgdW5pZm9ybUJ1ZmZlcnMgPSBbXG4gICAgbmV3IFVpbnQ4QXJyYXkoWzAsIDEsIDIsIDNdKS5idWZmZXIsXG4gICAgbmV3IFVpbnQ4QXJyYXkoWzQsIDUsIDYsIDddKS5idWZmZXIsXG4gICAgbmV3IFVpbnQ4QXJyYXkoWzgsIDksIDEwLCAxMV0pLmJ1ZmZlcixcbiAgICBuZXcgVWludDhBcnJheShbMTIsIDEzLCAxNCwgMTVdKS5idWZmZXIsXG4gICAgbmV3IFVpbnQ4QXJyYXkoWzE2XSkuYnVmZmVyLFxuICBdO1xuXG4gIGNvbnN0IG5vblVuaWZvcm1CdWZmZXJzID0gW1xuICAgIG5ldyBVaW50OEFycmF5KFswLCAxLCAyXSkuYnVmZmVyLFxuICAgIG5ldyBVaW50OEFycmF5KFszLCA0LCA1LCA2LCA3XSkuYnVmZmVyLFxuICAgIG5ldyBVaW50OEFycmF5KFs4LCA5LCAxMCwgMTFdKS5idWZmZXIsXG4gICAgbmV3IFVpbnQ4QXJyYXkoWzEyLCAxMywgMTQsIDE1LCAxNl0pLmJ1ZmZlcixcbiAgXTtcblxuICBjb25zdCBidWZmZXJUZXN0Q2FzZXMgPSBbXG4gICAgWyd1bmlmb3JtJywgdW5pZm9ybUJ1ZmZlcnNdLFxuICAgIFsnbm9uLXVuaWZvcm0nLCBub25Vbmlmb3JtQnVmZmVyc11cbiAgXSBhcyBjb25zdDtcblxuICBmb3IgKGNvbnN0IFtidWZmZXJzVHlwZSwgYnVmZmVyc10gb2YgYnVmZmVyVGVzdENhc2VzKSB7XG4gICAgbGV0IGNvbXBvc2l0ZTogQ29tcG9zaXRlQXJyYXlCdWZmZXI7XG4gICAgYmVmb3JlRWFjaCgoKSA9PiB7XG4gICAgICBjb21wb3NpdGUgPSBuZXcgQ29tcG9zaXRlQXJyYXlCdWZmZXIoYnVmZmVycyk7XG4gICAgfSk7XG5cbiAgICBpdChgJHtidWZmZXJzVHlwZX06IHNsaWNlcyBhY3Jvc3MgbXVsdGlwbGUgYnVmZmVyc2AsICgpID0+IHtcbiAgICAgIGV4cGVjdEFycmF5c0VxdWFsKG5ldyBVaW50OEFycmF5KGNvbXBvc2l0ZS5zbGljZSgxLCAxMykpLFxuICAgICAgICAgICAgICAgICAgICAgICAgWzEsMiwzLDQsNSw2LDcsOCw5LDEwLDExLDEyXSk7XG4gICAgfSk7XG5cbiAgICBpdChgJHtidWZmZXJzVHlwZX06IHNsaWNlcyB0byB0aGUgZW5kIG9mIHRoZSBhcnJheSB3aGVuIFxcJ2VuZFxcJyBpcyBub3QgYCArXG4gICAgICAnc3BlY2lmaWVkJywgKCkgPT4ge1xuICAgICAgICBleHBlY3RBcnJheXNFcXVhbChuZXcgVWludDhBcnJheShjb21wb3NpdGUuc2xpY2UoNSkpLFxuICAgICAgICAgICAgICAgICAgICAgICAgICBbNSw2LDcsOCw5LDEwLDExLDEyLDEzLDE0LDE1LDE2XSk7XG4gICAgICB9KTtcblxuICAgIGl0KGAke2J1ZmZlcnNUeXBlfTogbWFrZXMgYSBjb3B5IHdoZW4gc2xpY2UoKSBpcyBjYWxsZWQgd2l0aCBubyBhcmd1bWVudHNgLFxuICAgICAgICgpID0+IHtcbiAgICAgICAgIGV4cGVjdEFycmF5c0VxdWFsKG5ldyBVaW50OEFycmF5KGNvbXBvc2l0ZS5zbGljZSgpKSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIFswLDEsMiwzLDQsNSw2LDcsOCw5LDEwLDExLDEyLDEzLDE0LDE1LDE2XSk7XG4gICAgICAgfSk7XG5cbiAgICBpdChgJHtidWZmZXJzVHlwZX06IHNsaWNlcyBmcm9tIHplcm8gd2hlbiBzdGFydCBpcyBuZWdhdGl2ZWAsICgpID0+IHtcbiAgICAgIGV4cGVjdEFycmF5c0VxdWFsKG5ldyBVaW50OEFycmF5KGNvbXBvc2l0ZS5zbGljZSgtNCwgNSkpLFxuICAgICAgICAgICAgICAgICAgICAgICAgWzAsMSwyLDMsNF0pO1xuICAgIH0pO1xuXG4gICAgaXQoYCR7YnVmZmVyc1R5cGV9OiBzbGljZXMgdG8gdGhlIGVuZCB3aGVuIGVuZCBpcyBncmVhdGVyIHRoYW4gbGVuZ3RoYCxcbiAgICAgICAoKSA9PiB7XG4gICAgICAgICBleHBlY3RBcnJheXNFcXVhbChuZXcgVWludDhBcnJheShjb21wb3NpdGUuc2xpY2UoNywgMTAwMCkpLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgWzcsOCw5LDEwLDExLDEyLDEzLDE0LDE1LDE2XSk7XG4gICAgICAgfSk7XG5cbiAgICBpdChgJHtidWZmZXJzVHlwZX06IHNsaWNlcyBtdWx0aXBsZSByYW5nZXMgb3V0IG9mIG9yZGVyYCwgKCkgPT4ge1xuICAgICAgZXhwZWN0QXJyYXlzRXF1YWwobmV3IFVpbnQ4QXJyYXkoY29tcG9zaXRlLnNsaWNlKDEzLCAxNSkpLCBbMTMsIDE0XSk7XG4gICAgICBleHBlY3RBcnJheXNFcXVhbChuZXcgVWludDhBcnJheShjb21wb3NpdGUuc2xpY2UoMCwgMikpLCBbMCwgMV0pO1xuICAgICAgZXhwZWN0QXJyYXlzRXF1YWwobmV3IFVpbnQ4QXJyYXkoY29tcG9zaXRlLnNsaWNlKDksIDEzKSksXG4gICAgICAgICAgICAgICAgICAgICAgICBbOSwgMTAsIDExLCAxMl0pO1xuICAgIH0pO1xuICB9XG5cbiAgaXQoJ2NhbiBiZSBjcmVhdGVkIGZyb20gYW4gZW1wdHkgYXJyYXlidWZmZXInLCAoKSA9PiB7XG4gICAgY29uc3QgYXJyYXkgPSBuZXcgVWludDhBcnJheShbXSk7XG4gICAgY29uc3Qgc2luZ2xlQ29tcG9zaXRlID0gbmV3IENvbXBvc2l0ZUFycmF5QnVmZmVyKGFycmF5LmJ1ZmZlcik7XG4gICAgZXhwZWN0QXJyYXlzRXF1YWwobmV3IFVpbnQ4QXJyYXkoc2luZ2xlQ29tcG9zaXRlLnNsaWNlKCkpLCBbXSk7XG4gIH0pO1xuXG4gIGl0KCdjYW4gYmUgY3JlYXRlZCBmcm9tIGEgc2luZ2xlIGFycmF5JywgKCkgPT4ge1xuICAgIGNvbnN0IGFycmF5ID0gbmV3IFVpbnQ4QXJyYXkoWzEsMiwzXSk7XG4gICAgY29uc3Qgc2luZ2xlQ29tcG9zaXRlID0gbmV3IENvbXBvc2l0ZUFycmF5QnVmZmVyKGFycmF5LmJ1ZmZlcik7XG4gICAgZXhwZWN0QXJyYXlzRXF1YWwobmV3IFVpbnQ4QXJyYXkoc2luZ2xlQ29tcG9zaXRlLnNsaWNlKCkpLCBhcnJheSk7XG4gIH0pO1xuXG4gIGl0KCdjYW4gYmUgY3JlYXRlZCBmcm9tIHplcm8gYXJyYXlzJywgKCkgPT4ge1xuICAgIGNvbnN0IHNpbmdsZUNvbXBvc2l0ZSA9IG5ldyBDb21wb3NpdGVBcnJheUJ1ZmZlcihbXSk7XG4gICAgZXhwZWN0QXJyYXlzRXF1YWwobmV3IFVpbnQ4QXJyYXkoc2luZ2xlQ29tcG9zaXRlLnNsaWNlKCkpLFxuICAgICAgICAgICAgICAgICAgICAgIG5ldyBVaW50OEFycmF5KCkpO1xuICB9KTtcblxuICBpdCgnY2FuIGJlIGNyZWF0ZWQgZnJvbSB1bmRlZmluZWQgaW5wdXQnLCAoKSA9PiB7XG4gICAgY29uc3Qgc2luZ2xlQ29tcG9zaXRlID0gbmV3IENvbXBvc2l0ZUFycmF5QnVmZmVyKCk7XG4gICAgZXhwZWN0QXJyYXlzRXF1YWwobmV3IFVpbnQ4QXJyYXkoc2luZ2xlQ29tcG9zaXRlLnNsaWNlKCkpLFxuICAgICAgICAgICAgICAgICAgICAgIG5ldyBVaW50OEFycmF5KCkpO1xuICB9KTtcblxuICBpdCgndHJlYXRzIE5hTiBhcyB6ZXJvIHdoZW4gcGFzc2VkIGFzIHRoZSBzdGFydCBvZiBzbGljZScsICgpID0+IHtcbiAgICBjb25zdCBhcnJheSA9IG5ldyBVaW50OEFycmF5KFsxLDIsM10pO1xuICAgIGNvbnN0IGNvbXBvc2l0ZSA9IG5ldyBDb21wb3NpdGVBcnJheUJ1ZmZlcihhcnJheS5idWZmZXIpO1xuICAgIGV4cGVjdEFycmF5c0VxdWFsKG5ldyBVaW50OEFycmF5KGNvbXBvc2l0ZS5zbGljZShOYU4sIDIpKSwgWzEsMl0pO1xuICB9KTtcblxuICBpdCgndHJlYXRzIE5hTiBhcyB6ZXJvIHdoZW4gcGFzc2VkIGFzIHRoZSBlbmQgb2Ygc2xpY2UnLCAoKSA9PiB7XG4gICAgY29uc3QgYXJyYXkgPSBuZXcgVWludDhBcnJheShbMSwyLDNdKTtcbiAgICBjb25zdCBjb21wb3NpdGUgPSBuZXcgQ29tcG9zaXRlQXJyYXlCdWZmZXIoYXJyYXkuYnVmZmVyKTtcbiAgICBleHBlY3RBcnJheXNFcXVhbChuZXcgVWludDhBcnJheShjb21wb3NpdGUuc2xpY2UoMCwgTmFOKSksIFtdKTtcbiAgfSk7XG5cbiAgaXQoJ3N1cHBvcnRzIFR5cGVkQXJyYXkgaW5wdXQnLCAoKSA9PiB7XG4gICAgLy8gVGhpcyBzdXBwb3J0IGlzIG5lY2Vzc2FyeSBmb3Igc29tZSB0ZXN0cyBpbiB0ZmpzLWNvbnZlcnRlci4gTWF5YmUgdGhvc2VcbiAgICAvLyB0ZXN0cyBhcmUgbWlzY29uZmlndXJlZD9cbiAgICBjb25zdCBhcnJheSA9IG5ldyBVaW50OEFycmF5KFsxLDIsM10pO1xuICAgIGNvbnN0IGNvbXBvc2l0ZSA9IG5ldyBDb21wb3NpdGVBcnJheUJ1ZmZlcihhcnJheSk7XG4gICAgZXhwZWN0QXJyYXlzRXF1YWwobmV3IFVpbnQ4QXJyYXkoY29tcG9zaXRlLnNsaWNlKDAsIDIpKSwgWzEsMl0pO1xuICB9KTtcbn0pO1xuIl19