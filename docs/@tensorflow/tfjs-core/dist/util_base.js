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
/**
 * Shuffles the array in-place using Fisher-Yates algorithm.
 *
 * ```js
 * const a = [1, 2, 3, 4, 5];
 * tf.util.shuffle(a);
 * console.log(a);
 * ```
 *
 * @param array The array to shuffle in-place.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
// tslint:disable-next-line:no-any
export function shuffle(array) {
    let counter = array.length;
    let index = 0;
    // While there are elements in the array
    while (counter > 0) {
        // Pick a random index
        index = (Math.random() * counter) | 0;
        // Decrease counter by 1
        counter--;
        // And swap the last element with it
        swap(array, counter, index);
    }
}
/**
 * Shuffles two arrays in-place the same way using Fisher-Yates algorithm.
 *
 * ```js
 * const a = [1,2,3,4,5];
 * const b = [11,22,33,44,55];
 * tf.util.shuffleCombo(a, b);
 * console.log(a, b);
 * ```
 *
 * @param array The first array to shuffle in-place.
 * @param array2 The second array to shuffle in-place with the same permutation
 *     as the first array.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
export function shuffleCombo(
// tslint:disable-next-line:no-any
array, 
// tslint:disable-next-line:no-any
array2) {
    if (array.length !== array2.length) {
        throw new Error(`Array sizes must match to be shuffled together ` +
            `First array length was ${array.length}` +
            `Second array length was ${array2.length}`);
    }
    let counter = array.length;
    let index = 0;
    // While there are elements in the array
    while (counter > 0) {
        // Pick a random index
        index = (Math.random() * counter) | 0;
        // Decrease counter by 1
        counter--;
        // And swap the last element of each array with it
        swap(array, counter, index);
        swap(array2, counter, index);
    }
}
/** Clamps a value to a specified range. */
export function clamp(min, x, max) {
    return Math.max(min, Math.min(x, max));
}
export function nearestLargerEven(val) {
    return val % 2 === 0 ? val : val + 1;
}
export function swap(object, left, right) {
    const temp = object[left];
    object[left] = object[right];
    object[right] = temp;
}
export function sum(arr) {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
        sum += arr[i];
    }
    return sum;
}
/**
 * Returns a sample from a uniform [a, b) distribution.
 *
 * @param a The minimum support (inclusive).
 * @param b The maximum support (exclusive).
 * @return A pseudorandom number on the half-open interval [a,b).
 */
export function randUniform(a, b) {
    const r = Math.random();
    return (b * r) + (1 - r) * a;
}
/** Returns the squared Euclidean distance between two vectors. */
export function distSquared(a, b) {
    let result = 0;
    for (let i = 0; i < a.length; i++) {
        const diff = Number(a[i]) - Number(b[i]);
        result += diff * diff;
    }
    return result;
}
/**
 * Asserts that the expression is true. Otherwise throws an error with the
 * provided message.
 *
 * ```js
 * const x = 2;
 * tf.util.assert(x === 2, 'x is not 2');
 * ```
 *
 * @param expr The expression to assert (as a boolean).
 * @param msg A function that returns the message to report when throwing an
 *     error. We use a function for performance reasons.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
export function assert(expr, msg) {
    if (!expr) {
        throw new Error(typeof msg === 'string' ? msg : msg());
    }
}
export function assertShapesMatch(shapeA, shapeB, errorMessagePrefix = '') {
    assert(arraysEqual(shapeA, shapeB), () => errorMessagePrefix + ` Shapes ${shapeA} and ${shapeB} must match`);
}
export function assertNonNull(a) {
    assert(a != null, () => `The input to the tensor constructor must be a non-null value.`);
}
/**
 * Returns the size (number of elements) of the tensor given its shape.
 *
 * ```js
 * const shape = [3, 4, 2];
 * const size = tf.util.sizeFromShape(shape);
 * console.log(size);
 * ```
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
export function sizeFromShape(shape) {
    if (shape.length === 0) {
        // Scalar.
        return 1;
    }
    let size = shape[0];
    for (let i = 1; i < shape.length; i++) {
        size *= shape[i];
    }
    return size;
}
export function isScalarShape(shape) {
    return shape.length === 0;
}
export function arraysEqualWithNull(n1, n2) {
    if (n1 === n2) {
        return true;
    }
    if (n1 == null || n2 == null) {
        return false;
    }
    if (n1.length !== n2.length) {
        return false;
    }
    for (let i = 0; i < n1.length; i++) {
        if (n1[i] !== null && n2[i] !== null && n1[i] !== n2[i]) {
            return false;
        }
    }
    return true;
}
export function arraysEqual(n1, n2) {
    if (n1 === n2) {
        return true;
    }
    if (n1 == null || n2 == null) {
        return false;
    }
    if (n1.length !== n2.length) {
        return false;
    }
    for (let i = 0; i < n1.length; i++) {
        if (n1[i] !== n2[i]) {
            return false;
        }
    }
    return true;
}
export function isInt(a) {
    return a % 1 === 0;
}
export function tanh(x) {
    // tslint:disable-next-line:no-any
    if (Math.tanh != null) {
        // tslint:disable-next-line:no-any
        return Math.tanh(x);
    }
    if (x === Infinity) {
        return 1;
    }
    else if (x === -Infinity) {
        return -1;
    }
    else {
        const e2x = Math.exp(2 * x);
        return (e2x - 1) / (e2x + 1);
    }
}
export function sizeToSquarishShape(size) {
    const width = Math.ceil(Math.sqrt(size));
    return [width, Math.ceil(size / width)];
}
/**
 * Creates a new array with randomized indices to a given quantity.
 *
 * ```js
 * const randomTen = tf.util.createShuffledIndices(10);
 * console.log(randomTen);
 * ```
 *
 * @param number Quantity of how many shuffled indices to create.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
export function createShuffledIndices(n) {
    const shuffledIndices = new Uint32Array(n);
    for (let i = 0; i < n; ++i) {
        shuffledIndices[i] = i;
    }
    shuffle(shuffledIndices);
    return shuffledIndices;
}
export function rightPad(a, size) {
    if (size <= a.length) {
        return a;
    }
    return a + ' '.repeat(size - a.length);
}
export function repeatedTry(checkFn, delayFn = (counter) => 0, maxCounter, scheduleFn) {
    return new Promise((resolve, reject) => {
        let tryCount = 0;
        const tryFn = () => {
            if (checkFn()) {
                resolve();
                return;
            }
            tryCount++;
            const nextBackoff = delayFn(tryCount);
            if (maxCounter != null && tryCount >= maxCounter) {
                reject();
                return;
            }
            if (scheduleFn != null) {
                scheduleFn(tryFn, nextBackoff);
            }
            else {
                // google3 does not allow assigning another variable to setTimeout.
                // Don't refactor this so scheduleFn has a default value of setTimeout.
                setTimeout(tryFn, nextBackoff);
            }
        };
        tryFn();
    });
}
/**
 * Given the full size of the array and a shape that may contain -1 as the
 * implicit dimension, returns the inferred shape where -1 is replaced.
 * E.g. For shape=[2, -1, 3] and size=24, it will return [2, 4, 3].
 *
 * @param shape The shape, which may contain -1 in some dimension.
 * @param size The full size (number of elements) of the array.
 * @return The inferred shape where -1 is replaced with the inferred size.
 */
export function inferFromImplicitShape(shape, size) {
    let shapeProd = 1;
    let implicitIdx = -1;
    for (let i = 0; i < shape.length; ++i) {
        if (shape[i] >= 0) {
            shapeProd *= shape[i];
        }
        else if (shape[i] === -1) {
            if (implicitIdx !== -1) {
                throw Error(`Shapes can only have 1 implicit size. ` +
                    `Found -1 at dim ${implicitIdx} and dim ${i}`);
            }
            implicitIdx = i;
        }
        else if (shape[i] < 0) {
            throw Error(`Shapes can not be < 0. Found ${shape[i]} at dim ${i}`);
        }
    }
    if (implicitIdx === -1) {
        if (size > 0 && size !== shapeProd) {
            throw Error(`Size(${size}) must match the product of shape ${shape}`);
        }
        return shape;
    }
    if (shapeProd === 0) {
        throw Error(`Cannot infer the missing size in [${shape}] when ` +
            `there are 0 elements`);
    }
    if (size % shapeProd !== 0) {
        throw Error(`The implicit shape can't be a fractional number. ` +
            `Got ${size} / ${shapeProd}`);
    }
    const newShape = shape.slice();
    newShape[implicitIdx] = size / shapeProd;
    return newShape;
}
export function parseAxisParam(axis, shape) {
    const rank = shape.length;
    // Normalize input
    axis = axis == null ? shape.map((s, i) => i) : [].concat(axis);
    // Check for valid range
    assert(axis.every(ax => ax >= -rank && ax < rank), () => `All values in axis param must be in range [-${rank}, ${rank}) but ` +
        `got axis ${axis}`);
    // Check for only integers
    assert(axis.every(ax => isInt(ax)), () => `All values in axis param must be integers but ` +
        `got axis ${axis}`);
    // Handle negative axis.
    return axis.map(a => a < 0 ? rank + a : a);
}
/** Reduces the shape by removing all dimensions of shape 1. */
export function squeezeShape(shape, axis) {
    const newShape = [];
    const keptDims = [];
    const isEmptyArray = axis != null && Array.isArray(axis) && axis.length === 0;
    const axes = (axis == null || isEmptyArray) ?
        null :
        parseAxisParam(axis, shape).sort();
    let j = 0;
    for (let i = 0; i < shape.length; ++i) {
        if (axes != null) {
            if (axes[j] === i && shape[i] !== 1) {
                throw new Error(`Can't squeeze axis ${i} since its dim '${shape[i]}' is not 1`);
            }
            if ((axes[j] == null || axes[j] > i) && shape[i] === 1) {
                newShape.push(shape[i]);
                keptDims.push(i);
            }
            if (axes[j] <= i) {
                j++;
            }
        }
        if (shape[i] !== 1) {
            newShape.push(shape[i]);
            keptDims.push(i);
        }
    }
    return { newShape, keptDims };
}
export function getTypedArrayFromDType(dtype, size) {
    return getArrayFromDType(dtype, size);
}
export function getArrayFromDType(dtype, size) {
    let values = null;
    if (dtype == null || dtype === 'float32') {
        values = new Float32Array(size);
    }
    else if (dtype === 'int32') {
        values = new Int32Array(size);
    }
    else if (dtype === 'bool') {
        values = new Uint8Array(size);
    }
    else if (dtype === 'string') {
        values = new Array(size);
    }
    else {
        throw new Error(`Unknown data type ${dtype}`);
    }
    return values;
}
export function checkConversionForErrors(vals, dtype) {
    for (let i = 0; i < vals.length; i++) {
        const num = vals[i];
        if (isNaN(num) || !isFinite(num)) {
            throw Error(`A tensor of type ${dtype} being uploaded contains ${num}.`);
        }
    }
}
/** Returns true if the dtype is valid. */
export function isValidDtype(dtype) {
    return dtype === 'bool' || dtype === 'complex64' || dtype === 'float32' ||
        dtype === 'int32' || dtype === 'string';
}
/**
 * Returns true if the new type can't encode the old type without loss of
 * precision.
 */
export function hasEncodingLoss(oldType, newType) {
    if (newType === 'complex64') {
        return false;
    }
    if (newType === 'float32' && oldType !== 'complex64') {
        return false;
    }
    if (newType === 'int32' && oldType !== 'float32' && oldType !== 'complex64') {
        return false;
    }
    if (newType === 'bool' && oldType === 'bool') {
        return false;
    }
    return true;
}
export function bytesPerElement(dtype) {
    if (dtype === 'float32' || dtype === 'int32') {
        return 4;
    }
    else if (dtype === 'complex64') {
        return 8;
    }
    else if (dtype === 'bool') {
        return 1;
    }
    else {
        throw new Error(`Unknown dtype ${dtype}`);
    }
}
/**
 * Returns the approximate number of bytes allocated in the string array - 2
 * bytes per character. Computing the exact bytes for a native string in JS
 * is not possible since it depends on the encoding of the html page that
 * serves the website.
 */
export function bytesFromStringArray(arr) {
    if (arr == null) {
        return 0;
    }
    let bytes = 0;
    arr.forEach(x => bytes += x.length);
    return bytes;
}
/** Returns true if the value is a string. */
export function isString(value) {
    return typeof value === 'string' || value instanceof String;
}
export function isBoolean(value) {
    return typeof value === 'boolean';
}
export function isNumber(value) {
    return typeof value === 'number';
}
export function inferDtype(values) {
    if (Array.isArray(values)) {
        return inferDtype(values[0]);
    }
    if (values instanceof Float32Array) {
        return 'float32';
    }
    else if (values instanceof Int32Array || values instanceof Uint8Array ||
        values instanceof Uint8ClampedArray) {
        return 'int32';
    }
    else if (isNumber(values)) {
        return 'float32';
    }
    else if (isString(values)) {
        return 'string';
    }
    else if (isBoolean(values)) {
        return 'bool';
    }
    return 'float32';
}
export function isFunction(f) {
    return !!(f && f.constructor && f.call && f.apply);
}
export function nearestDivisor(size, start) {
    for (let i = start; i < size; ++i) {
        if (size % i === 0) {
            return i;
        }
    }
    return size;
}
export function computeStrides(shape) {
    const rank = shape.length;
    if (rank < 2) {
        return [];
    }
    // Last dimension has implicit stride of 1, thus having D-1 (instead of D)
    // strides.
    const strides = new Array(rank - 1);
    strides[rank - 2] = shape[rank - 1];
    for (let i = rank - 3; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}
function createNestedArray(offset, shape, a, isComplex = false) {
    const ret = new Array();
    if (shape.length === 1) {
        const d = shape[0] * (isComplex ? 2 : 1);
        for (let i = 0; i < d; i++) {
            ret[i] = a[offset + i];
        }
    }
    else {
        const d = shape[0];
        const rest = shape.slice(1);
        const len = rest.reduce((acc, c) => acc * c) * (isComplex ? 2 : 1);
        for (let i = 0; i < d; i++) {
            ret[i] = createNestedArray(offset + i * len, rest, a, isComplex);
        }
    }
    return ret;
}
// Provide a nested array of TypedArray in given shape.
export function toNestedArray(shape, a, isComplex = false) {
    if (shape.length === 0) {
        // Scalar type should return a single number.
        return a[0];
    }
    const size = shape.reduce((acc, c) => acc * c) * (isComplex ? 2 : 1);
    if (size === 0) {
        // A tensor with shape zero should be turned into empty list.
        return [];
    }
    if (size !== a.length) {
        throw new Error(`[${shape}] does not match the input size ${a.length}${isComplex ? ' for a complex tensor' : ''}.`);
    }
    return createNestedArray(0, shape, a, isComplex);
}
export function convertBackendValuesAndArrayBuffer(data, dtype) {
    // If is type Uint8Array[], return it directly.
    if (Array.isArray(data)) {
        return data;
    }
    if (dtype === 'float32') {
        return data instanceof Float32Array ? data : new Float32Array(data);
    }
    else if (dtype === 'int32') {
        return data instanceof Int32Array ? data : new Int32Array(data);
    }
    else if (dtype === 'bool' || dtype === 'string') {
        return Uint8Array.from(new Int32Array(data));
    }
    else {
        throw new Error(`Unknown dtype ${dtype}`);
    }
}
export function makeOnesTypedArray(size, dtype) {
    const array = makeZerosTypedArray(size, dtype);
    for (let i = 0; i < array.length; i++) {
        array[i] = 1;
    }
    return array;
}
export function makeZerosTypedArray(size, dtype) {
    if (dtype == null || dtype === 'float32' || dtype === 'complex64') {
        return new Float32Array(size);
    }
    else if (dtype === 'int32') {
        return new Int32Array(size);
    }
    else if (dtype === 'bool') {
        return new Uint8Array(size);
    }
    else {
        throw new Error(`Unknown data type ${dtype}`);
    }
}
/**
 * Make nested `TypedArray` filled with zeros.
 * @param shape The shape information for the nested array.
 * @param dtype dtype of the array element.
 */
export function makeZerosNestedTypedArray(shape, dtype) {
    const size = shape.reduce((prev, curr) => prev * curr, 1);
    if (dtype == null || dtype === 'float32') {
        return toNestedArray(shape, new Float32Array(size));
    }
    else if (dtype === 'int32') {
        return toNestedArray(shape, new Int32Array(size));
    }
    else if (dtype === 'bool') {
        return toNestedArray(shape, new Uint8Array(size));
    }
    else {
        throw new Error(`Unknown data type ${dtype}`);
    }
}
export function assertNonNegativeIntegerDimensions(shape) {
    shape.forEach(dimSize => {
        assert(Number.isInteger(dimSize) && dimSize >= 0, () => `Tensor must have a shape comprised of positive integers but got ` +
            `shape [${shape}].`);
    });
}
/**
 * Computes flat index for a given location (multidimentionsal index) in a
 * Tensor/multidimensional array.
 *
 * @param locs Location in the tensor.
 * @param rank Rank of the tensor.
 * @param strides Tensor strides.
 */
export function locToIndex(locs, rank, strides) {
    if (rank === 0) {
        return 0;
    }
    else if (rank === 1) {
        return locs[0];
    }
    let index = locs[locs.length - 1];
    for (let i = 0; i < locs.length - 1; ++i) {
        index += strides[i] * locs[i];
    }
    return index;
}
/**
 * Computes the location (multidimensional index) in a
 * tensor/multidimentional array for a given flat index.
 *
 * @param index Index in flat array.
 * @param rank Rank of tensor.
 * @param strides Strides of tensor.
 */
export function indexToLoc(index, rank, strides) {
    if (rank === 0) {
        return [];
    }
    else if (rank === 1) {
        return [index];
    }
    const locs = new Array(rank);
    for (let i = 0; i < locs.length - 1; ++i) {
        locs[i] = Math.floor(index / strides[i]);
        index -= locs[i] * strides[i];
    }
    locs[locs.length - 1] = index;
    return locs;
}
/**
 * This method asserts whether an object is a Promise instance.
 * @param object
 */
// tslint:disable-next-line: no-any
export function isPromise(object) {
    //  We chose to not use 'obj instanceOf Promise' for two reasons:
    //  1. It only reliably works for es6 Promise, not other Promise
    //  implementations.
    //  2. It doesn't work with framework that uses zone.js. zone.js monkey
    //  patch the async calls, so it is possible the obj (patched) is
    //  comparing to a pre-patched Promise.
    return object && object.then && typeof object.then === 'function';
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidXRpbF9iYXNlLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy91dGlsX2Jhc2UudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBSUg7Ozs7Ozs7Ozs7OztHQVlHO0FBQ0gsa0NBQWtDO0FBQ2xDLE1BQU0sVUFBVSxPQUFPLENBQUMsS0FDWTtJQUNsQyxJQUFJLE9BQU8sR0FBRyxLQUFLLENBQUMsTUFBTSxDQUFDO0lBQzNCLElBQUksS0FBSyxHQUFHLENBQUMsQ0FBQztJQUNkLHdDQUF3QztJQUN4QyxPQUFPLE9BQU8sR0FBRyxDQUFDLEVBQUU7UUFDbEIsc0JBQXNCO1FBQ3RCLEtBQUssR0FBRyxDQUFDLElBQUksQ0FBQyxNQUFNLEVBQUUsR0FBRyxPQUFPLENBQUMsR0FBRyxDQUFDLENBQUM7UUFDdEMsd0JBQXdCO1FBQ3hCLE9BQU8sRUFBRSxDQUFDO1FBQ1Ysb0NBQW9DO1FBQ3BDLElBQUksQ0FBQyxLQUFLLEVBQUUsT0FBTyxFQUFFLEtBQUssQ0FBQyxDQUFDO0tBQzdCO0FBQ0gsQ0FBQztBQUVEOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUNILE1BQU0sVUFBVSxZQUFZO0FBQ3hCLGtDQUFrQztBQUNsQyxLQUFnRDtBQUNoRCxrQ0FBa0M7QUFDbEMsTUFBaUQ7SUFDbkQsSUFBSSxLQUFLLENBQUMsTUFBTSxLQUFLLE1BQU0sQ0FBQyxNQUFNLEVBQUU7UUFDbEMsTUFBTSxJQUFJLEtBQUssQ0FDWCxpREFBaUQ7WUFDakQsMEJBQTBCLEtBQUssQ0FBQyxNQUFNLEVBQUU7WUFDeEMsMkJBQTJCLE1BQU0sQ0FBQyxNQUFNLEVBQUUsQ0FBQyxDQUFDO0tBQ2pEO0lBQ0QsSUFBSSxPQUFPLEdBQUcsS0FBSyxDQUFDLE1BQU0sQ0FBQztJQUMzQixJQUFJLEtBQUssR0FBRyxDQUFDLENBQUM7SUFDZCx3Q0FBd0M7SUFDeEMsT0FBTyxPQUFPLEdBQUcsQ0FBQyxF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QUFJLEtBQUssQ0FBQyxJQUFJLENBQUMsQ0FBQztJQUN2QyxLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsSUFBSSxDQUFDLE1BQU0sR0FBRyxDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUU7UUFDeEMsSUFBSSxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxLQUFLLENBQUMsS0FBSyxHQUFHLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3pDLEtBQUssSUFBSSxJQUFJLENBQUMsQ0FBQyxDQUFDLEdBQUcsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDO0tBQy9CO0lBQ0QsSUFBSSxDQUFDLElBQUksQ0FBQyxNQUFNLEdBQUcsQ0FBQyxDQUFDLEdBQUcsS0FBSyxDQUFDO0lBQzlCLE9BQU8sSUFBSSxDQUFDO0FBQ2QsQ0FBQztBQUVEOzs7R0FHRztBQUNILG1DQUFtQztBQUNuQyxNQUFNLFVBQVUsU0FBUyxDQUFDLE1BQVc7SUFDbkMsaUVBQWlFO0lBQ2pFLGdFQUFnRTtJQUNoRSxvQkFBb0I7SUFDcEIsdUVBQXVFO0lBQ3ZFLGlFQUFpRTtJQUNqRSx1Q0FBdUM7SUFDdkMsT0FBTyxNQUFNLElBQUksTUFBTSxDQUFDLElBQUksSUFBSSxPQUFPLE1BQU0sQ0FBQyxJQUFJLEtBQUssVUFBVSxDQUFDO0FBQ3BFLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7QmFja2VuZFZhbHVlcywgRGF0YVR5cGUsIERhdGFUeXBlTWFwLCBGbGF0VmVjdG9yLCBOdW1lcmljRGF0YVR5cGUsIFRlbnNvckxpa2UsIFR5cGVkQXJyYXksIFdlYkdMRGF0YSwgV2ViR1BVRGF0YX0gZnJvbSAnLi90eXBlcyc7XG5cbi8qKlxuICogU2h1ZmZsZXMgdGhlIGFycmF5IGluLXBsYWNlIHVzaW5nIEZpc2hlci1ZYXRlcyBhbGdvcml0aG0uXG4gKlxuICogYGBganNcbiAqIGNvbnN0IGEgPSBbMSwgMiwgMywgNCwgNV07XG4gKiB0Zi51dGlsLnNodWZmbGUoYSk7XG4gKiBjb25zb2xlLmxvZyhhKTtcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSBhcnJheSBUaGUgYXJyYXkgdG8gc2h1ZmZsZSBpbi1wbGFjZS5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnVXRpbCcsIG5hbWVzcGFjZTogJ3V0aWwnfVxuICovXG4vLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG5leHBvcnQgZnVuY3Rpb24gc2h1ZmZsZShhcnJheTogYW55W118VWludDMyQXJyYXl8SW50MzJBcnJheXxcbiAgICAgICAgICAgICAgICAgICAgICAgIEZsb2F0MzJBcnJheSk6IHZvaWQge1xuICBsZXQgY291bnRlciA9IGFycmF5Lmxlbmd0aDtcbiAgbGV0IGluZGV4ID0gMDtcbiAgLy8gV2hpbGUgdGhlcmUgYXJlIGVsZW1lbnRzIGluIHRoZSBhcnJheVxuICB3aGlsZSAoY291bnRlciA+IDApIHtcbiAgICAvLyBQaWNrIGEgcmFuZG9tIGluZGV4XG4gICAgaW5kZXggPSAoTWF0aC5yYW5kb20oKSAqIGNvdW50ZXIpIHwgMDtcbiAgICAvLyBEZWNyZWFzZSBjb3VudGVyIGJ5IDFcbiAgICBjb3VudGVyLS07XG4gICAgLy8gQW5kIHN3YXAgdGhlIGxhc3QgZWxlbWVudCB3aXRoIGl0XG4gICAgc3dhcChhcnJheSwgY291bnRlciwgaW5kZXgpO1xuICB9XG59XG5cbi8qKlxuICogU2h1ZmZsZXMgdHdvIGFycmF5cyBpbi1wbGFjZSB0aGUgc2FtZSB3YXkgdXNpbmcgRmlzaGVyLVlhdGVzIGFsZ29yaXRobS5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgYSA9IFsxLDIsMyw0LDVdO1xuICogY29uc3QgYiA9IFsxMSwyMiwzMyw0NCw1NV07XG4gKiB0Zi51dGlsLnNodWZmbGVDb21ibyhhLCBiKTtcbiAqIGNvbnNvbGUubG9nKGEsIGIpO1xuICogYGBgXG4gKlxuICogQHBhcmFtIGFycmF5IFRoZSBmaXJzdCBhcnJheSB0byBzaHVmZmxlIGluLXBsYWNlLlxuICogQHBhcmFtIGFycmF5MiBUaGUgc2Vjb25kIGFycmF5IHRvIHNodWZmbGUgaW4tcGxhY2Ugd2l0aCB0aGUgc2FtZSBwZXJtdXRhdGlvblxuICogICAgIGFzIHRoZSBmaXJzdCBhcnJheS5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnVXRpbCcsIG5hbWVzcGFjZTogJ3V0aWwnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gc2h1ZmZsZUNvbWJvKFxuICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICBhcnJheTogYW55W118VWludDMyQXJyYXl8SW50MzJBcnJheXxGbG9hdDMyQXJyYXksXG4gICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgIGFycmF5MjogYW55W118VWludDMyQXJyYXl8SW50MzJBcnJheXxGbG9hdDMyQXJyYXkpOiB2b2lkIHtcbiAgaWYgKGFycmF5Lmxlbmd0aCAhPT0gYXJyYXkyLmxlbmd0aCkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgYEFycmF5IHNpemVzIG11c3QgbWF0Y2ggdG8gYmUgc2h1ZmZsZWQgdG9nZXRoZXIgYCArXG4gICAgICAgIGBGaXJzdCBhcnJheSBsZW5ndGggd2FzICR7YXJyYXkubGVuZ3RofWAgK1xuICAgICAgICBgU2Vjb25kIGFycmF5IGxlbmd0aCB3YXMgJHthcnJheTIubGVuZ3RofWApO1xuICB9XG4gIGxldCBjb3VudGVyID0gYXJyYXkubGVuZ3RoO1xuICBsZXQgaW5kZXggPSAwO1xuICAvLyBXaGlsZSB0aGVyZSBhcmUgZWxlbWVudHMgaW4gdGhlIGFycmF5XG4gIHdoaWxlIChjb3VudGVyID4gMCkge1xuICAgIC8vIFBpY2sgYSByYW5kb20gaW5kZXhcbiAgICBpbmRleCA9IChNYXRoLnJhbmRvbSgpICogY291bnRlcikgfCAwO1xuICAgIC8vIERlY3JlYXNlIGNvdW50ZXIgYnkgMVxuICAgIGNvdW50ZXItLTtcbiAgICAvLyBBbmQgc3dhcCB0aGUgbGFzdCBlbGVtZW50IG9mIGVhY2ggYXJyYXkgd2l0aCBpdFxuICAgIHN3YXAoYXJyYXksIGNvdW50ZXIsIGluZGV4KTtcbiAgICBzd2FwKGFycmF5MiwgY291bnRlciwgaW5kZXgpO1xuICB9XG59XG5cbi8qKiBDbGFtcHMgYSB2YWx1ZSB0byBhIHNwZWNpZmllZCByYW5nZS4gKi9cbmV4cG9ydCBmdW5jdGlvbiBjbGFtcChtaW46IG51bWJlciwgeDogbnVtYmVyLCBtYXg6IG51bWJlcik6IG51bWJlciB7XG4gIHJldHVybiBNYXRoLm1heChtaW4sIE1hdGgubWluKHgsIG1heCkpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gbmVhcmVzdExhcmdlckV2ZW4odmFsOiBudW1iZXIpOiBudW1iZXIge1xuICByZXR1cm4gdmFsICUgMiA9PT0gMCA/IHZhbCA6IHZhbCArIDE7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBzd2FwPFQ+KFxuICAgIG9iamVjdDoge1tpbmRleDogbnVtYmVyXTogVH0sIGxlZnQ6IG51bWJlciwgcmlnaHQ6IG51bWJlcikge1xuICBjb25zdCB0ZW1wID0gb2JqZWN0W2xlZnRdO1xuICBvYmplY3RbbGVmdF0gPSBvYmplY3RbcmlnaHRdO1xuICBvYmplY3RbcmlnaHRdID0gdGVtcDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHN1bShhcnI6IG51bWJlcltdKTogbnVtYmVyIHtcbiAgbGV0IHN1bSA9IDA7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgYXJyLmxlbmd0aDsgaSsrKSB7XG4gICAgc3VtICs9IGFycltpXTtcbiAgfVxuICByZXR1cm4gc3VtO1xufVxuXG4vKipcbiAqIFJldHVybnMgYSBzYW1wbGUgZnJvbSBhIHVuaWZvcm0gW2EsIGIpIGRpc3RyaWJ1dGlvbi5cbiAqXG4gKiBAcGFyYW0gYSBUaGUgbWluaW11bSBzdXBwb3J0IChpbmNsdXNpdmUpLlxuICogQHBhcmFtIGIgVGhlIG1heGltdW0gc3VwcG9ydCAoZXhjbHVzaXZlKS5cbiAqIEByZXR1cm4gQSBwc2V1ZG9yYW5kb20gbnVtYmVyIG9uIHRoZSBoYWxmLW9wZW4gaW50ZXJ2YWwgW2EsYikuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiByYW5kVW5pZm9ybShhOiBudW1iZXIsIGI6IG51bWJlcikge1xuICBjb25zdCByID0gTWF0aC5yYW5kb20oKTtcbiAgcmV0dXJuIChiICogcikgKyAoMSAtIHIpICogYTtcbn1cblxuLyoqIFJldHVybnMgdGhlIHNxdWFyZWQgRXVjbGlkZWFuIGRpc3RhbmNlIGJldHdlZW4gdHdvIHZlY3RvcnMuICovXG5leHBvcnQgZnVuY3Rpb24gZGlzdFNxdWFyZWQoYTogRmxhdFZlY3RvciwgYjogRmxhdFZlY3Rvcik6IG51bWJlciB7XG4gIGxldCByZXN1bHQgPSAwO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IGEubGVuZ3RoOyBpKyspIHtcbiAgICBjb25zdCBkaWZmID0gTnVtYmVyKGFbaV0pIC0gTnVtYmVyKGJbaV0pO1xuICAgIHJlc3VsdCArPSBkaWZmICogZGlmZjtcbiAgfVxuICByZXR1cm4gcmVzdWx0O1xufVxuXG4vKipcbiAqIEFzc2VydHMgdGhhdCB0aGUgZXhwcmVzc2lvbiBpcyB0cnVlLiBPdGhlcndpc2UgdGhyb3dzIGFuIGVycm9yIHdpdGggdGhlXG4gKiBwcm92aWRlZCBtZXNzYWdlLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gMjtcbiAqIHRmLnV0aWwuYXNzZXJ0KHggPT09IDIsICd4IGlzIG5vdCAyJyk7XG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0gZXhwciBUaGUgZXhwcmVzc2lvbiB0byBhc3NlcnQgKGFzIGEgYm9vbGVhbikuXG4gKiBAcGFyYW0gbXNnIEEgZnVuY3Rpb24gdGhhdCByZXR1cm5zIHRoZSBtZXNzYWdlIHRvIHJlcG9ydCB3aGVuIHRocm93aW5nIGFuXG4gKiAgICAgZXJyb3IuIFdlIHVzZSBhIGZ1bmN0aW9uIGZvciBwZXJmb3JtYW5jZSByZWFzb25zLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdVdGlsJywgbmFtZXNwYWNlOiAndXRpbCd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBhc3NlcnQoZXhwcjogYm9vbGVhbiwgbXNnOiAoKSA9PiBzdHJpbmcpIHtcbiAgaWYgKCFleHByKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKHR5cGVvZiBtc2cgPT09ICdzdHJpbmcnID8gbXNnIDogbXNnKCkpO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBhc3NlcnRTaGFwZXNNYXRjaChcbiAgICBzaGFwZUE6IG51bWJlcltdLCBzaGFwZUI6IG51bWJlcltdLCBlcnJvck1lc3NhZ2VQcmVmaXggPSAnJyk6IHZvaWQge1xuICBhc3NlcnQoXG4gICAgICBhcnJheXNFcXVhbChzaGFwZUEsIHNoYXBlQiksXG4gICAgICAoKSA9PiBlcnJvck1lc3NhZ2VQcmVmaXggKyBgIFNoYXBlcyAke3NoYXBlQX0gYW5kICR7c2hhcGVCfSBtdXN0IG1hdGNoYCk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBhc3NlcnROb25OdWxsKGE6IFRlbnNvckxpa2UpOiB2b2lkIHtcbiAgYXNzZXJ0KFxuICAgICAgYSAhPSBudWxsLFxuICAgICAgKCkgPT4gYFRoZSBpbnB1dCB0byB0aGUgdGVuc29yIGNvbnN0cnVjdG9yIG11c3QgYmUgYSBub24tbnVsbCB2YWx1ZS5gKTtcbn1cblxuLyoqXG4gKiBSZXR1cm5zIHRoZSBzaXplIChudW1iZXIgb2YgZWxlbWVudHMpIG9mIHRoZSB0ZW5zb3IgZ2l2ZW4gaXRzIHNoYXBlLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBzaGFwZSA9IFszLCA0LCAyXTtcbiAqIGNvbnN0IHNpemUgPSB0Zi51dGlsLnNpemVGcm9tU2hhcGUoc2hhcGUpO1xuICogY29uc29sZS5sb2coc2l6ZSk7XG4gKiBgYGBcbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnVXRpbCcsIG5hbWVzcGFjZTogJ3V0aWwnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gc2l6ZUZyb21TaGFwZShzaGFwZTogbnVtYmVyW10pOiBudW1iZXIge1xuICBpZiAoc2hhcGUubGVuZ3RoID09PSAwKSB7XG4gICAgLy8gU2NhbGFyLlxuICAgIHJldHVybiAxO1xuICB9XG4gIGxldCBzaXplID0gc2hhcGVbMF07XG4gIGZvciAobGV0IGkgPSAxOyBpIDwgc2hhcGUubGVuZ3RoOyBpKyspIHtcbiAgICBzaXplICo9IHNoYXBlW2ldO1xuICB9XG4gIHJldHVybiBzaXplO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gaXNTY2FsYXJTaGFwZShzaGFwZTogbnVtYmVyW10pOiBib29sZWFuIHtcbiAgcmV0dXJuIHNoYXBlLmxlbmd0aCA9PT0gMDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGFycmF5c0VxdWFsV2l0aE51bGwobjE6IG51bWJlcltdLCBuMjogbnVtYmVyW10pIHtcbiAgaWYgKG4xID09PSBuMikge1xuICAgIHJldHVybiB0cnVlO1xuICB9XG5cbiAgaWYgKG4xID09IG51bGwgfHwgbjIgPT0gbnVsbCkge1xuICAgIHJldHVybiBmYWxzZTtcbiAgfVxuXG4gIGlmIChuMS5sZW5ndGggIT09IG4yLmxlbmd0aCkge1xuICAgIHJldHVybiBmYWxzZTtcbiAgfVxuXG4gIGZvciAobGV0IGkgPSAwOyBpIDwgbjEubGVuZ3RoOyBpKyspIHtcbiAgICBpZiAobjFbaV0gIT09IG51bGwgJiYgbjJbaV0gIT09IG51bGwgJiYgbjFbaV0gIT09IG4yW2ldKSB7XG4gICAgICByZXR1cm4gZmFsc2U7XG4gICAgfVxuICB9XG4gIHJldHVybiB0cnVlO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYXJyYXlzRXF1YWwobjE6IEZsYXRWZWN0b3IsIG4yOiBGbGF0VmVjdG9yKSB7XG4gIGlmIChuMSA9PT0gbjIpIHtcbiAgICByZXR1cm4gdHJ1ZTtcbiAgfVxuICBpZiAobjEgPT0gbnVsbCB8fCBuMiA9PSBudWxsKSB7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG5cbiAgaWYgKG4xLmxlbmd0aCAhPT0gbjIubGVuZ3RoKSB7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgbjEubGVuZ3RoOyBpKyspIHtcbiAgICBpZiAobjFbaV0gIT09IG4yW2ldKSB7XG4gICAgICByZXR1cm4gZmFsc2U7XG4gICAgfVxuICB9XG4gIHJldHVybiB0cnVlO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gaXNJbnQoYTogbnVtYmVyKTogYm9vbGVhbiB7XG4gIHJldHVybiBhICUgMSA9PT0gMDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHRhbmgoeDogbnVtYmVyKTogbnVtYmVyIHtcbiAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICBpZiAoKE1hdGggYXMgYW55KS50YW5oICE9IG51bGwpIHtcbiAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgcmV0dXJuIChNYXRoIGFzIGFueSkudGFuaCh4KTtcbiAgfVxuICBpZiAoeCA9PT0gSW5maW5pdHkpIHtcbiAgICByZXR1cm4gMTtcbiAgfSBlbHNlIGlmICh4ID09PSAtSW5maW5pdHkpIHtcbiAgICByZXR1cm4gLTE7XG4gIH0gZWxzZSB7XG4gICAgY29uc3QgZTJ4ID0gTWF0aC5leHAoMiAqIHgpO1xuICAgIHJldHVybiAoZTJ4IC0gMSkgLyAoZTJ4ICsgMSk7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHNpemVUb1NxdWFyaXNoU2hhcGUoc2l6ZTogbnVtYmVyKTogW251bWJlciwgbnVtYmVyXSB7XG4gIGNvbnN0IHdpZHRoID0gTWF0aC5jZWlsKE1hdGguc3FydChzaXplKSk7XG4gIHJldHVybiBbd2lkdGgsIE1hdGguY2VpbChzaXplIC8gd2lkdGgpXTtcbn1cblxuLyoqXG4gKiBDcmVhdGVzIGEgbmV3IGFycmF5IHdpdGggcmFuZG9taXplZCBpbmRpY2VzIHRvIGEgZ2l2ZW4gcXVhbnRpdHkuXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHJhbmRvbVRlbiA9IHRmLnV0aWwuY3JlYXRlU2h1ZmZsZWRJbmRpY2VzKDEwKTtcbiAqIGNvbnNvbGUubG9nKHJhbmRvbVRlbik7XG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0gbnVtYmVyIFF1YW50aXR5IG9mIGhvdyBtYW55IHNodWZmbGVkIGluZGljZXMgdG8gY3JlYXRlLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdVdGlsJywgbmFtZXNwYWNlOiAndXRpbCd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVTaHVmZmxlZEluZGljZXMobjogbnVtYmVyKTogVWludDMyQXJyYXkge1xuICBjb25zdCBzaHVmZmxlZEluZGljZXMgPSBuZXcgVWludDMyQXJyYXkobik7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgbjsgKytpKSB7XG4gICAgc2h1ZmZsZWRJbmRpY2VzW2ldID0gaTtcbiAgfVxuICBzaHVmZmxlKHNodWZmbGVkSW5kaWNlcyk7XG4gIHJldHVybiBzaHVmZmxlZEluZGljZXM7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiByaWdodFBhZChhOiBzdHJpbmcsIHNpemU6IG51bWJlcik6IHN0cmluZyB7XG4gIGlmIChzaXplIDw9IGEubGVuZ3RoKSB7XG4gICAgcmV0dXJuIGE7XG4gIH1cbiAgcmV0dXJuIGEgKyAnICcucmVwZWF0KHNpemUgLSBhLmxlbmd0aCk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiByZXBlYXRlZFRyeShcbiAgICBjaGVja0ZuOiAoKSA9PiBib29sZWFuLCBkZWxheUZuID0gKGNvdW50ZXI6IG51bWJlcikgPT4gMCxcbiAgICBtYXhDb3VudGVyPzogbnVtYmVyLFxuICAgIHNjaGVkdWxlRm4/OiAoZnVuY3Rpb25SZWY6IEZ1bmN0aW9uLCBkZWxheTogbnVtYmVyKSA9PlxuICAgICAgICB2b2lkKTogUHJvbWlzZTx2b2lkPiB7XG4gIHJldHVybiBuZXcgUHJvbWlzZTx2b2lkPigocmVzb2x2ZSwgcmVqZWN0KSA9PiB7XG4gICAgbGV0IHRyeUNvdW50ID0gMDtcblxuICAgIGNvbnN0IHRyeUZuID0gKCkgPT4ge1xuICAgICAgaWYgKGNoZWNrRm4oKSkge1xuICAgICAgICByZXNvbHZlKCk7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cblxuICAgICAgdHJ5Q291bnQrKztcblxuICAgICAgY29uc3QgbmV4dEJhY2tvZmYgPSBkZWxheUZuKHRyeUNvdW50KTtcblxuICAgICAgaWYgKG1heENvdW50ZXIgIT0gbnVsbCAmJiB0cnlDb3VudCA+PSBtYXhDb3VudGVyKSB7XG4gICAgICAgIHJlamVjdCgpO1xuICAgICAgICByZXR1cm47XG4gICAgICB9XG5cbiAgICAgIGlmIChzY2hlZHVsZUZuICE9IG51bGwpIHtcbiAgICAgICAgc2NoZWR1bGVGbih0cnlGbiwgbmV4dEJhY2tvZmYpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgLy8gZ29vZ2xlMyBkb2VzIG5vdCBhbGxvdyBhc3NpZ25pbmcgYW5vdGhlciB2YXJpYWJsZSB0byBzZXRUaW1lb3V0LlxuICAgICAgICAvLyBEb24ndCByZWZhY3RvciB0aGlzIHNvIHNjaGVkdWxlRm4gaGFzIGEgZGVmYXVsdCB2YWx1ZSBvZiBzZXRUaW1lb3V0LlxuICAgICAgICBzZXRUaW1lb3V0KHRyeUZuLCBuZXh0QmFja29mZik7XG4gICAgICB9XG4gICAgfTtcblxuICAgIHRyeUZuKCk7XG4gIH0pO1xufVxuXG4vKipcbiAqIEdpdmVuIHRoZSBmdWxsIHNpemUgb2YgdGhlIGFycmF5IGFuZCBhIHNoYXBlIHRoYXQgbWF5IGNvbnRhaW4gLTEgYXMgdGhlXG4gKiBpbXBsaWNpdCBkaW1lbnNpb24sIHJldHVybnMgdGhlIGluZmVycmVkIHNoYXBlIHdoZXJlIC0xIGlzIHJlcGxhY2VkLlxuICogRS5nLiBGb3Igc2hhcGU9WzIsIC0xLCAzXSBhbmQgc2l6ZT0yNCwgaXQgd2lsbCByZXR1cm4gWzIsIDQsIDNdLlxuICpcbiAqIEBwYXJhbSBzaGFwZSBUaGUgc2hhcGUsIHdoaWNoIG1heSBjb250YWluIC0xIGluIHNvbWUgZGltZW5zaW9uLlxuICogQHBhcmFtIHNpemUgVGhlIGZ1bGwgc2l6ZSAobnVtYmVyIG9mIGVsZW1lbnRzKSBvZiB0aGUgYXJyYXkuXG4gKiBAcmV0dXJuIFRoZSBpbmZlcnJlZCBzaGFwZSB3aGVyZSAtMSBpcyByZXBsYWNlZCB3aXRoIHRoZSBpbmZlcnJlZCBzaXplLlxuICovXG5leHBvcnQgZnVuY3Rpb24gaW5mZXJGcm9tSW1wbGljaXRTaGFwZShcbiAgICBzaGFwZTogbnVtYmVyW10sIHNpemU6IG51bWJlcik6IG51bWJlcltdIHtcbiAgbGV0IHNoYXBlUHJvZCA9IDE7XG4gIGxldCBpbXBsaWNpdElkeCA9IC0xO1xuXG4gIGZvciAobGV0IGkgPSAwOyBpIDwgc2hhcGUubGVuZ3RoOyArK2kpIHtcbiAgICBpZiAoc2hhcGVbaV0gPj0gMCkge1xuICAgICAgc2hhcGVQcm9kICo9IHNoYXBlW2ldO1xuICAgIH0gZWxzZSBpZiAoc2hhcGVbaV0gPT09IC0xKSB7XG4gICAgICBpZiAoaW1wbGljaXRJZHggIT09IC0xKSB7XG4gICAgICAgIHRocm93IEVycm9yKFxuICAgICAgICAgICAgYFNoYXBlcyBjYW4gb25seSBoYXZlIDEgaW1wbGljaXQgc2l6ZS4gYCArXG4gICAgICAgICAgICBgRm91bmQgLTEgYXQgZGltICR7aW1wbGljaXRJZHh9IGFuZCBkaW0gJHtpfWApO1xuICAgICAgfVxuICAgICAgaW1wbGljaXRJZHggPSBpO1xuICAgIH0gZWxzZSBpZiAoc2hhcGVbaV0gPCAwKSB7XG4gICAgICB0aHJvdyBFcnJvcihgU2hhcGVzIGNhbiBub3QgYmUgPCAwLiBGb3VuZCAke3NoYXBlW2ldfSBhdCBkaW0gJHtpfWApO1xuICAgIH1cbiAgfVxuXG4gIGlmIChpbXBsaWNpdElkeCA9PT0gLTEpIHtcbiAgICBpZiAoc2l6ZSA+IDAgJiYgc2l6ZSAhPT0gc2hhcGVQcm9kKSB7XG4gICAgICB0aHJvdyBFcnJvcihgU2l6ZSgke3NpemV9KSBtdXN0IG1hdGNoIHRoZSBwcm9kdWN0IG9mIHNoYXBlICR7c2hhcGV9YCk7XG4gICAgfVxuICAgIHJldHVybiBzaGFwZTtcbiAgfVxuXG4gIGlmIChzaGFwZVByb2QgPT09IDApIHtcbiAgICB0aHJvdyBFcnJvcihcbiAgICAgICAgYENhbm5vdCBpbmZlciB0aGUgbWlzc2luZyBzaXplIGluIFske3NoYXBlfV0gd2hlbiBgICtcbiAgICAgICAgYHRoZXJlIGFyZSAwIGVsZW1lbnRzYCk7XG4gIH1cbiAgaWYgKHNpemUgJSBzaGFwZVByb2QgIT09IDApIHtcbiAgICB0aHJvdyBFcnJvcihcbiAgICAgICAgYFRoZSBpbXBsaWNpdCBzaGFwZSBjYW4ndCBiZSBhIGZyYWN0aW9uYWwgbnVtYmVyLiBgICtcbiAgICAgICAgYEdvdCAke3NpemV9IC8gJHtzaGFwZVByb2R9YCk7XG4gIH1cblxuICBjb25zdCBuZXdTaGFwZSA9IHNoYXBlLnNsaWNlKCk7XG4gIG5ld1NoYXBlW2ltcGxpY2l0SWR4XSA9IHNpemUgLyBzaGFwZVByb2Q7XG4gIHJldHVybiBuZXdTaGFwZTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHBhcnNlQXhpc1BhcmFtKFxuICAgIGF4aXM6IG51bWJlcnxudW1iZXJbXSwgc2hhcGU6IG51bWJlcltdKTogbnVtYmVyW10ge1xuICBjb25zdCByYW5rID0gc2hhcGUubGVuZ3RoO1xuXG4gIC8vIE5vcm1hbGl6ZSBpbnB1dFxuICBheGlzID0gYXhpcyA9PSBudWxsID8gc2hhcGUubWFwKChzLCBpKSA9PiBpKSA6IFtdLmNvbmNhdChheGlzKTtcblxuICAvLyBDaGVjayBmb3IgdmFsaWQgcmFuZ2VcbiAgYXNzZXJ0KFxuICAgICAgYXhpcy5ldmVyeShheCA9PiBheCA+PSAtcmFuayAmJiBheCA8IHJhbmspLFxuICAgICAgKCkgPT5cbiAgICAgICAgICBgQWxsIHZhbHVlcyBpbiBheGlzIHBhcmFtIG11c3QgYmUgaW4gcmFuZ2UgWy0ke3Jhbmt9LCAke3Jhbmt9KSBidXQgYCArXG4gICAgICAgICAgYGdvdCBheGlzICR7YXhpc31gKTtcblxuICAvLyBDaGVjayBmb3Igb25seSBpbnRlZ2Vyc1xuICBhc3NlcnQoXG4gICAgICBheGlzLmV2ZXJ5KGF4ID0+IGlzSW50KGF4KSksXG4gICAgICAoKSA9PiBgQWxsIHZhbHVlcyBpbiBheGlzIHBhcmFtIG11c3QgYmUgaW50ZWdlcnMgYnV0IGAgK1xuICAgICAgICAgIGBnb3QgYXhpcyAke2F4aXN9YCk7XG5cbiAgLy8gSGFuZGxlIG5lZ2F0aXZlIGF4aXMuXG4gIHJldHVybiBheGlzLm1hcChhID0+IGEgPCAwID8gcmFuayArIGEgOiBhKTtcbn1cblxuLyoqIFJlZHVjZXMgdGhlIHNoYXBlIGJ5IHJlbW92aW5nIGFsbCBkaW1lbnNpb25zIG9mIHNoYXBlIDEuICovXG5leHBvcnQgZnVuY3Rpb24gc3F1ZWV6ZVNoYXBlKHNoYXBlOiBudW1iZXJbXSwgYXhpcz86IG51bWJlcltdKTpcbiAgICB7bmV3U2hhcGU6IG51bWJlcltdLCBrZXB0RGltczogbnVtYmVyW119IHtcbiAgY29uc3QgbmV3U2hhcGU6IG51bWJlcltdID0gW107XG4gIGNvbnN0IGtlcHREaW1zOiBudW1iZXJbXSA9IFtdO1xuICBjb25zdCBpc0VtcHR5QXJyYXkgPSBheGlzICE9IG51bGwgJiYgQXJyYXkuaXNBcnJheShheGlzKSAmJiBheGlzLmxlbmd0aCA9PT0gMDtcbiAgY29uc3QgYXhlcyA9IChheGlzID09IG51bGwgfHwgaXNFbXB0eUFycmF5KSA/XG4gICAgICBudWxsIDpcbiAgICAgIHBhcnNlQXhpc1BhcmFtKGF4aXMsIHNoYXBlKS5zb3J0KCk7XG4gIGxldCBqID0gMDtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBzaGFwZS5sZW5ndGg7ICsraSkge1xuICAgIGlmIChheGVzICE9IG51bGwpIHtcbiAgICAgIGlmIChheGVzW2pdID09PSBpICYmIHNoYXBlW2ldICE9PSAxKSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgIGBDYW4ndCBzcXVlZXplIGF4aXMgJHtpfSBzaW5jZSBpdHMgZGltICcke3NoYXBlW2ldfScgaXMgbm90IDFgKTtcbiAgICAgIH1cbiAgICAgIGlmICgoYXhlc1tqXSA9PSBudWxsIHx8IGF4ZXNbal0gPiBpKSAmJiBzaGFwZVtpXSA9PT0gMSkge1xuICAgICAgICBuZXdTaGFwZS5wdXNoKHNoYXBlW2ldKTtcbiAgICAgICAga2VwdERpbXMucHVzaChpKTtcbiAgICAgIH1cbiAgICAgIGlmIChheGVzW2pdIDw9IGkpIHtcbiAgICAgICAgaisrO1xuICAgICAgfVxuICAgIH1cbiAgICBpZiAoc2hhcGVbaV0gIT09IDEpIHtcbiAgICAgIG5ld1NoYXBlLnB1c2goc2hhcGVbaV0pO1xuICAgICAga2VwdERpbXMucHVzaChpKTtcbiAgICB9XG4gIH1cbiAgcmV0dXJuIHtuZXdTaGFwZSwga2VwdERpbXN9O1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0VHlwZWRBcnJheUZyb21EVHlwZTxEIGV4dGVuZHMgTnVtZXJpY0RhdGFUeXBlPihcbiAgICBkdHlwZTogRCwgc2l6ZTogbnVtYmVyKTogRGF0YVR5cGVNYXBbRF0ge1xuICByZXR1cm4gZ2V0QXJyYXlGcm9tRFR5cGU8RD4oZHR5cGUsIHNpemUpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0QXJyYXlGcm9tRFR5cGU8RCBleHRlbmRzIERhdGFUeXBlPihcbiAgICBkdHlwZTogRCwgc2l6ZTogbnVtYmVyKTogRGF0YVR5cGVNYXBbRF0ge1xuICBsZXQgdmFsdWVzID0gbnVsbDtcbiAgaWYgKGR0eXBlID09IG51bGwgfHwgZHR5cGUgPT09ICdmbG9hdDMyJykge1xuICAgIHZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkoc2l6ZSk7XG4gIH0gZWxzZSBpZiAoZHR5cGUgPT09ICdpbnQzMicpIHtcbiAgICB2YWx1ZXMgPSBuZXcgSW50MzJBcnJheShzaXplKTtcbiAgfSBlbHNlIGlmIChkdHlwZSA9PT0gJ2Jvb2wnKSB7XG4gICAgdmFsdWVzID0gbmV3IFVpbnQ4QXJyYXkoc2l6ZSk7XG4gIH0gZWxzZSBpZiAoZHR5cGUgPT09ICdzdHJpbmcnKSB7XG4gICAgdmFsdWVzID0gbmV3IEFycmF5PHN0cmluZz4oc2l6ZSk7XG4gIH0gZWxzZSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGBVbmtub3duIGRhdGEgdHlwZSAke2R0eXBlfWApO1xuICB9XG4gIHJldHVybiB2YWx1ZXMgYXMgRGF0YVR5cGVNYXBbRF07XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjaGVja0NvbnZlcnNpb25Gb3JFcnJvcnM8RCBleHRlbmRzIERhdGFUeXBlPihcbiAgICB2YWxzOiBEYXRhVHlwZU1hcFtEXXxudW1iZXJbXSwgZHR5cGU6IEQpOiB2b2lkIHtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWxzLmxlbmd0aDsgaSsrKSB7XG4gICAgY29uc3QgbnVtID0gdmFsc1tpXSBhcyBudW1iZXI7XG4gICAgaWYgKGlzTmFOKG51bSkgfHwgIWlzRmluaXRlKG51bSkpIHtcbiAgICAgIHRocm93IEVycm9yKGBBIHRlbnNvciBvZiB0eXBlICR7ZHR5cGV9IGJlaW5nIHVwbG9hZGVkIGNvbnRhaW5zICR7bnVtfS5gKTtcbiAgICB9XG4gIH1cbn1cblxuLyoqIFJldHVybnMgdHJ1ZSBpZiB0aGUgZHR5cGUgaXMgdmFsaWQuICovXG5leHBvcnQgZnVuY3Rpb24gaXNWYWxpZER0eXBlKGR0eXBlOiBEYXRhVHlwZSk6IGJvb2xlYW4ge1xuICByZXR1cm4gZHR5cGUgPT09ICdib29sJyB8fCBkdHlwZSA9PT0gJ2NvbXBsZXg2NCcgfHwgZHR5cGUgPT09ICdmbG9hdDMyJyB8fFxuICAgICAgZHR5cGUgPT09ICdpbnQzMicgfHwgZHR5cGUgPT09ICdzdHJpbmcnO1xufVxuXG4vKipcbiAqIFJldHVybnMgdHJ1ZSBpZiB0aGUgbmV3IHR5cGUgY2FuJ3QgZW5jb2RlIHRoZSBvbGQgdHlwZSB3aXRob3V0IGxvc3Mgb2ZcbiAqIHByZWNpc2lvbi5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGhhc0VuY29kaW5nTG9zcyhvbGRUeXBlOiBEYXRhVHlwZSwgbmV3VHlwZTogRGF0YVR5cGUpOiBib29sZWFuIHtcbiAgaWYgKG5ld1R5cGUgPT09ICdjb21wbGV4NjQnKSB7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG4gIGlmIChuZXdUeXBlID09PSAnZmxvYXQzMicgJiYgb2xkVHlwZSAhPT0gJ2NvbXBsZXg2NCcpIHtcbiAgICByZXR1cm4gZmFsc2U7XG4gIH1cbiAgaWYgKG5ld1R5cGUgPT09ICdpbnQzMicgJiYgb2xkVHlwZSAhPT0gJ2Zsb2F0MzInICYmIG9sZFR5cGUgIT09ICdjb21wbGV4NjQnKSB7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG4gIGlmIChuZXdUeXBlID09PSAnYm9vbCcgJiYgb2xkVHlwZSA9PT0gJ2Jvb2wnKSB7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG4gIHJldHVybiB0cnVlO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYnl0ZXNQZXJFbGVtZW50KGR0eXBlOiBEYXRhVHlwZSk6IG51bWJlciB7XG4gIGlmIChkdHlwZSA9PT0gJ2Zsb2F0MzInIHx8IGR0eXBlID09PSAnaW50MzInKSB7XG4gICAgcmV0dXJuIDQ7XG4gIH0gZWxzZSBpZiAoZHR5cGUgPT09ICdjb21wbGV4NjQnKSB7XG4gICAgcmV0dXJuIDg7XG4gIH0gZWxzZSBpZiAoZHR5cGUgPT09ICdib29sJykge1xuICAgIHJldHVybiAxO1xuICB9IGVsc2Uge1xuICAgIHRocm93IG5ldyBFcnJvcihgVW5rbm93biBkdHlwZSAke2R0eXBlfWApO1xuICB9XG59XG5cbi8qKlxuICogUmV0dXJucyB0aGUgYXBwcm94aW1hdGUgbnVtYmVyIG9mIGJ5dGVzIGFsbG9jYXRlZCBpbiB0aGUgc3RyaW5nIGFycmF5IC0gMlxuICogYnl0ZXMgcGVyIGNoYXJhY3Rlci4gQ29tcHV0aW5nIHRoZSBleGFjdCBieXRlcyBmb3IgYSBuYXRpdmUgc3RyaW5nIGluIEpTXG4gKiBpcyBub3QgcG9zc2libGUgc2luY2UgaXQgZGVwZW5kcyBvbiB0aGUgZW5jb2Rpbmcgb2YgdGhlIGh0bWwgcGFnZSB0aGF0XG4gKiBzZXJ2ZXMgdGhlIHdlYnNpdGUuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBieXRlc0Zyb21TdHJpbmdBcnJheShhcnI6IFVpbnQ4QXJyYXlbXSk6IG51bWJlciB7XG4gIGlmIChhcnIgPT0gbnVsbCkge1xuICAgIHJldHVybiAwO1xuICB9XG4gIGxldCBieXRlcyA9IDA7XG4gIGFyci5mb3JFYWNoKHggPT4gYnl0ZXMgKz0geC5sZW5ndGgpO1xuICByZXR1cm4gYnl0ZXM7XG59XG5cbi8qKiBSZXR1cm5zIHRydWUgaWYgdGhlIHZhbHVlIGlzIGEgc3RyaW5nLiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGlzU3RyaW5nKHZhbHVlOiB7fSk6IHZhbHVlIGlzIHN0cmluZyB7XG4gIHJldHVybiB0eXBlb2YgdmFsdWUgPT09ICdzdHJpbmcnIHx8IHZhbHVlIGluc3RhbmNlb2YgU3RyaW5nO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gaXNCb29sZWFuKHZhbHVlOiB7fSk6IGJvb2xlYW4ge1xuICByZXR1cm4gdHlwZW9mIHZhbHVlID09PSAnYm9vbGVhbic7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBpc051bWJlcih2YWx1ZToge30pOiBib29sZWFuIHtcbiAgcmV0dXJuIHR5cGVvZiB2YWx1ZSA9PT0gJ251bWJlcic7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBpbmZlckR0eXBlKHZhbHVlczogVGVuc29yTGlrZXxXZWJHTERhdGF8V2ViR1BVRGF0YSk6IERhdGFUeXBlIHtcbiAgaWYgKEFycmF5LmlzQXJyYXkodmFsdWVzKSkge1xuICAgIHJldHVybiBpbmZlckR0eXBlKHZhbHVlc1swXSk7XG4gIH1cbiAgaWYgKHZhbHVlcyBpbnN0YW5jZW9mIEZsb2F0MzJBcnJheSkge1xuICAgIHJldHVybiAnZmxvYXQzMic7XG4gIH0gZWxzZSBpZiAoXG4gICAgICB2YWx1ZXMgaW5zdGFuY2VvZiBJbnQzMkFycmF5IHx8IHZhbHVlcyBpbnN0YW5jZW9mIFVpbnQ4QXJyYXkgfHxcbiAgICAgIHZhbHVlcyBpbnN0YW5jZW9mIFVpbnQ4Q2xhbXBlZEFycmF5KSB7XG4gICAgcmV0dXJuICdpbnQzMic7XG4gIH0gZWxzZSBpZiAoaXNOdW1iZXIodmFsdWVzKSkge1xuICAgIHJldHVybiAnZmxvYXQzMic7XG4gIH0gZWxzZSBpZiAoaXNTdHJpbmcodmFsdWVzKSkge1xuICAgIHJldHVybiAnc3RyaW5nJztcbiAgfSBlbHNlIGlmIChpc0Jvb2xlYW4odmFsdWVzKSkge1xuICAgIHJldHVybiAnYm9vbCc7XG4gIH1cbiAgcmV0dXJuICdmbG9hdDMyJztcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGlzRnVuY3Rpb24oZjogRnVuY3Rpb24pIHtcbiAgcmV0dXJuICEhKGYgJiYgZi5jb25zdHJ1Y3RvciAmJiBmLmNhbGwgJiYgZi5hcHBseSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBuZWFyZXN0RGl2aXNvcihzaXplOiBudW1iZXIsIHN0YXJ0OiBudW1iZXIpOiBudW1iZXIge1xuICBmb3IgKGxldCBpID0gc3RhcnQ7IGkgPCBzaXplOyArK2kpIHtcbiAgICBpZiAoc2l6ZSAlIGkgPT09IDApIHtcbiAgICAgIHJldHVybiBpO1xuICAgIH1cbiAgfVxuICByZXR1cm4gc2l6ZTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNvbXB1dGVTdHJpZGVzKHNoYXBlOiBudW1iZXJbXSk6IG51bWJlcltdIHtcbiAgY29uc3QgcmFuayA9IHNoYXBlLmxlbmd0aDtcbiAgaWYgKHJhbmsgPCAyKSB7XG4gICAgcmV0dXJuIFtdO1xuICB9XG5cbiAgLy8gTGFzdCBkaW1lbnNpb24gaGFzIGltcGxpY2l0IHN0cmlkZSBvZiAxLCB0aHVzIGhhdmluZyBELTEgKGluc3RlYWQgb2YgRClcbiAgLy8gc3RyaWRlcy5cbiAgY29uc3Qgc3RyaWRlcyA9IG5ldyBBcnJheShyYW5rIC0gMSk7XG4gIHN0cmlkZXNbcmFuayAtIDJdID0gc2hhcGVbcmFuayAtIDFdO1xuICBmb3IgKGxldCBpID0gcmFuayAtIDM7IGkgPj0gMDsgLS1pKSB7XG4gICAgc3RyaWRlc1tpXSA9IHN0cmlkZXNbaSArIDFdICogc2hhcGVbaSArIDFdO1xuICB9XG4gIHJldHVybiBzdHJpZGVzO1xufVxuXG5mdW5jdGlvbiBjcmVhdGVOZXN0ZWRBcnJheShcbiAgICBvZmZzZXQ6IG51bWJlciwgc2hhcGU6IG51bWJlcltdLCBhOiBUeXBlZEFycmF5LCBpc0NvbXBsZXggPSBmYWxzZSkge1xuICBjb25zdCByZXQgPSBuZXcgQXJyYXkoKTtcbiAgaWYgKHNoYXBlLmxlbmd0aCA9PT0gMSkge1xuICAgIGNvbnN0IGQgPSBzaGFwZVswXSAqIChpc0NvbXBsZXggPyAyIDogMSk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBkOyBpKyspIHtcbiAgICAgIHJldFtpXSA9IGFbb2Zmc2V0ICsgaV07XG4gICAgfVxuICB9IGVsc2Uge1xuICAgIGNvbnN0IGQgPSBzaGFwZVswXTtcbiAgICBjb25zdCByZXN0ID0gc2hhcGUuc2xpY2UoMSk7XG4gICAgY29uc3QgbGVuID0gcmVzdC5yZWR1Y2UoKGFjYywgYykgPT4gYWNjICogYykgKiAoaXNDb21wbGV4ID8gMiA6IDEpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgZDsgaSsrKSB7XG4gICAgICByZXRbaV0gPSBjcmVhdGVOZXN0ZWRBcnJheShvZmZzZXQgKyBpICogbGVuLCByZXN0LCBhLCBpc0NvbXBsZXgpO1xuICAgIH1cbiAgfVxuICByZXR1cm4gcmV0O1xufVxuXG4vLyBQcm92aWRlIGEgbmVzdGVkIGFycmF5IG9mIFR5cGVkQXJyYXkgaW4gZ2l2ZW4gc2hhcGUuXG5leHBvcnQgZnVuY3Rpb24gdG9OZXN0ZWRBcnJheShcbiAgICBzaGFwZTogbnVtYmVyW10sIGE6IFR5cGVkQXJyYXksIGlzQ29tcGxleCA9IGZhbHNlKSB7XG4gIGlmIChzaGFwZS5sZW5ndGggPT09IDApIHtcbiAgICAvLyBTY2FsYXIgdHlwZSBzaG91bGQgcmV0dXJuIGEgc2luZ2xlIG51bWJlci5cbiAgICByZXR1cm4gYVswXTtcbiAgfVxuICBjb25zdCBzaXplID0gc2hhcGUucmVkdWNlKChhY2MsIGMpID0+IGFjYyAqIGMpICogKGlzQ29tcGxleCA/IDIgOiAxKTtcbiAgaWYgKHNpemUgPT09IDApIHtcbiAgICAvLyBBIHRlbnNvciB3aXRoIHNoYXBlIHplcm8gc2hvdWxkIGJlIHR1cm5lZCBpbnRvIGVtcHR5IGxpc3QuXG4gICAgcmV0dXJuIFtdO1xuICB9XG4gIGlmIChzaXplICE9PSBhLmxlbmd0aCkge1xuICAgIHRocm93IG5ldyBFcnJvcihgWyR7c2hhcGV9XSBkb2VzIG5vdCBtYXRjaCB0aGUgaW5wdXQgc2l6ZSAke2EubGVuZ3RofSR7XG4gICAgICAgIGlzQ29tcGxleCA/ICcgZm9yIGEgY29tcGxleCB0ZW5zb3InIDogJyd9LmApO1xuICB9XG5cbiAgcmV0dXJuIGNyZWF0ZU5lc3RlZEFycmF5KDAsIHNoYXBlLCBhLCBpc0NvbXBsZXgpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY29udmVydEJhY2tlbmRWYWx1ZXNBbmRBcnJheUJ1ZmZlcihcbiAgICBkYXRhOiBCYWNrZW5kVmFsdWVzfEFycmF5QnVmZmVyLCBkdHlwZTogRGF0YVR5cGUpIHtcbiAgLy8gSWYgaXMgdHlwZSBVaW50OEFycmF5W10sIHJldHVybiBpdCBkaXJlY3RseS5cbiAgaWYgKEFycmF5LmlzQXJyYXkoZGF0YSkpIHtcbiAgICByZXR1cm4gZGF0YTtcbiAgfVxuICBpZiAoZHR5cGUgPT09ICdmbG9hdDMyJykge1xuICAgIHJldHVybiBkYXRhIGluc3RhbmNlb2YgRmxvYXQzMkFycmF5ID8gZGF0YSA6IG5ldyBGbG9hdDMyQXJyYXkoZGF0YSk7XG4gIH0gZWxzZSBpZiAoZHR5cGUgPT09ICdpbnQzMicpIHtcbiAgICByZXR1cm4gZGF0YSBpbnN0YW5jZW9mIEludDMyQXJyYXkgPyBkYXRhIDogbmV3IEludDMyQXJyYXkoZGF0YSk7XG4gIH0gZWxzZSBpZiAoZHR5cGUgPT09ICdib29sJyB8fCBkdHlwZSA9PT0gJ3N0cmluZycpIHtcbiAgICByZXR1cm4gVWludDhBcnJheS5mcm9tKG5ldyBJbnQzMkFycmF5KGRhdGEpKTtcbiAgfSBlbHNlIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYFVua25vd24gZHR5cGUgJHtkdHlwZX1gKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gbWFrZU9uZXNUeXBlZEFycmF5PEQgZXh0ZW5kcyBEYXRhVHlwZT4oXG4gICAgc2l6ZTogbnVtYmVyLCBkdHlwZTogRCk6IERhdGFUeXBlTWFwW0RdIHtcbiAgY29uc3QgYXJyYXkgPSBtYWtlWmVyb3NUeXBlZEFycmF5KHNpemUsIGR0eXBlKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBhcnJheS5sZW5ndGg7IGkrKykge1xuICAgIGFycmF5W2ldID0gMTtcbiAgfVxuICByZXR1cm4gYXJyYXk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBtYWtlWmVyb3NUeXBlZEFycmF5PEQgZXh0ZW5kcyBEYXRhVHlwZT4oXG4gICAgc2l6ZTogbnVtYmVyLCBkdHlwZTogRCk6IERhdGFUeXBlTWFwW0RdIHtcbiAgaWYgKGR0eXBlID09IG51bGwgfHwgZHR5cGUgPT09ICdmbG9hdDMyJyB8fCBkdHlwZSA9PT0gJ2NvbXBsZXg2NCcpIHtcbiAgICByZXR1cm4gbmV3IEZsb2F0MzJBcnJheShzaXplKSBhcyBEYXRhVHlwZU1hcFtEXTtcbiAgfSBlbHNlIGlmIChkdHlwZSA9PT0gJ2ludDMyJykge1xuICAgIHJldHVybiBuZXcgSW50MzJBcnJheShzaXplKSBhcyBEYXRhVHlwZU1hcFtEXTtcbiAgfSBlbHNlIGlmIChkdHlwZSA9PT0gJ2Jvb2wnKSB7XG4gICAgcmV0dXJuIG5ldyBVaW50OEFycmF5KHNpemUpIGFzIERhdGFUeXBlTWFwW0RdO1xuICB9IGVsc2Uge1xuICAgIHRocm93IG5ldyBFcnJvcihgVW5rbm93biBkYXRhIHR5cGUgJHtkdHlwZX1gKTtcbiAgfVxufVxuXG4vKipcbiAqIE1ha2UgbmVzdGVkIGBUeXBlZEFycmF5YCBmaWxsZWQgd2l0aCB6ZXJvcy5cbiAqIEBwYXJhbSBzaGFwZSBUaGUgc2hhcGUgaW5mb3JtYXRpb24gZm9yIHRoZSBuZXN0ZWQgYXJyYXkuXG4gKiBAcGFyYW0gZHR5cGUgZHR5cGUgb2YgdGhlIGFycmF5IGVsZW1lbnQuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBtYWtlWmVyb3NOZXN0ZWRUeXBlZEFycmF5PEQgZXh0ZW5kcyBEYXRhVHlwZT4oXG4gICAgc2hhcGU6IG51bWJlcltdLCBkdHlwZTogRCkge1xuICBjb25zdCBzaXplID0gc2hhcGUucmVkdWNlKChwcmV2LCBjdXJyKSA9PiBwcmV2ICogY3VyciwgMSk7XG4gIGlmIChkdHlwZSA9PSBudWxsIHx8IGR0eXBlID09PSAnZmxvYXQzMicpIHtcbiAgICByZXR1cm4gdG9OZXN0ZWRBcnJheShzaGFwZSwgbmV3IEZsb2F0MzJBcnJheShzaXplKSk7XG4gIH0gZWxzZSBpZiAoZHR5cGUgPT09ICdpbnQzMicpIHtcbiAgICByZXR1cm4gdG9OZXN0ZWRBcnJheShzaGFwZSwgbmV3IEludDMyQXJyYXkoc2l6ZSkpO1xuICB9IGVsc2UgaWYgKGR0eXBlID09PSAnYm9vbCcpIHtcbiAgICByZXR1cm4gdG9OZXN0ZWRBcnJheShzaGFwZSwgbmV3IFVpbnQ4QXJyYXkoc2l6ZSkpO1xuICB9IGVsc2Uge1xuICAgIHRocm93IG5ldyBFcnJvcihgVW5rbm93biBkYXRhIHR5cGUgJHtkdHlwZX1gKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gYXNzZXJ0Tm9uTmVnYXRpdmVJbnRlZ2VyRGltZW5zaW9ucyhzaGFwZTogbnVtYmVyW10pIHtcbiAgc2hhcGUuZm9yRWFjaChkaW1TaXplID0+IHtcbiAgICBhc3NlcnQoXG4gICAgICAgIE51bWJlci5pc0ludGVnZXIoZGltU2l6ZSkgJiYgZGltU2l6ZSA+PSAwLFxuICAgICAgICAoKSA9PlxuICAgICAgICAgICAgYFRlbnNvciBtdXN0IGhhdmUgYSBzaGFwZSBjb21wcmlzZWQgb2YgcG9zaXRpdmUgaW50ZWdlcnMgYnV0IGdvdCBgICtcbiAgICAgICAgICAgIGBzaGFwZSBbJHtzaGFwZX1dLmApO1xuICB9KTtcbn1cblxuLyoqXG4gKiBDb21wdXRlcyBmbGF0IGluZGV4IGZvciBhIGdpdmVuIGxvY2F0aW9uIChtdWx0aWRpbWVudGlvbnNhbCBpbmRleCkgaW4gYVxuICogVGVuc29yL211bHRpZGltZW5zaW9uYWwgYXJyYXkuXG4gKlxuICogQHBhcmFtIGxvY3MgTG9jYXRpb24gaW4gdGhlIHRlbnNvci5cbiAqIEBwYXJhbSByYW5rIFJhbmsgb2YgdGhlIHRlbnNvci5cbiAqIEBwYXJhbSBzdHJpZGVzIFRlbnNvciBzdHJpZGVzLlxuICovXG5leHBvcnQgZnVuY3Rpb24gbG9jVG9JbmRleChcbiAgICBsb2NzOiBudW1iZXJbXSwgcmFuazogbnVtYmVyLCBzdHJpZGVzOiBudW1iZXJbXSk6IG51bWJlciB7XG4gIGlmIChyYW5rID09PSAwKSB7XG4gICAgcmV0dXJuIDA7XG4gIH0gZWxzZSBpZiAocmFuayA9PT0gMSkge1xuICAgIHJldHVybiBsb2NzWzBdO1xuICB9XG4gIGxldCBpbmRleCA9IGxvY3NbbG9jcy5sZW5ndGggLSAxXTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBsb2NzLmxlbmd0aCAtIDE7ICsraSkge1xuICAgIGluZGV4ICs9IHN0cmlkZXNbaV0gKiBsb2NzW2ldO1xuICB9XG4gIHJldHVybiBpbmRleDtcbn1cblxuLyoqXG4gKiBDb21wdXRlcyB0aGUgbG9jYXRpb24gKG11bHRpZGltZW5zaW9uYWwgaW5kZXgpIGluIGFcbiAqIHRlbnNvci9tdWx0aWRpbWVudGlvbmFsIGFycmF5IGZvciBhIGdpdmVuIGZsYXQgaW5kZXguXG4gKlxuICogQHBhcmFtIGluZGV4IEluZGV4IGluIGZsYXQgYXJyYXkuXG4gKiBAcGFyYW0gcmFuayBSYW5rIG9mIHRlbnNvci5cbiAqIEBwYXJhbSBzdHJpZGVzIFN0cmlkZXMgb2YgdGVuc29yLlxuICovXG5leHBvcnQgZnVuY3Rpb24gaW5kZXhUb0xvYyhcbiAgICBpbmRleDogbnVtYmVyLCByYW5rOiBudW1iZXIsIHN0cmlkZXM6IG51bWJlcltdKTogbnVtYmVyW10ge1xuICBpZiAocmFuayA9PT0gMCkge1xuICAgIHJldHVybiBbXTtcbiAgfSBlbHNlIGlmIChyYW5rID09PSAxKSB7XG4gICAgcmV0dXJuIFtpbmRleF07XG4gIH1cbiAgY29uc3QgbG9jczogbnVtYmVyW10gPSBuZXcgQXJyYXkocmFuayk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgbG9jcy5sZW5ndGggLSAxOyArK2kpIHtcbiAgICBsb2NzW2ldID0gTWF0aC5mbG9vcihpbmRleCAvIHN0cmlkZXNbaV0pO1xuICAgIGluZGV4IC09IGxvY3NbaV0gKiBzdHJpZGVzW2ldO1xuICB9XG4gIGxvY3NbbG9jcy5sZW5ndGggLSAxXSA9IGluZGV4O1xuICByZXR1cm4gbG9jcztcbn1cblxuLyoqXG4gKiBUaGlzIG1ldGhvZCBhc3NlcnRzIHdoZXRoZXIgYW4gb2JqZWN0IGlzIGEgUHJvbWlzZSBpbnN0YW5jZS5cbiAqIEBwYXJhbSBvYmplY3RcbiAqL1xuLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOiBuby1hbnlcbmV4cG9ydCBmdW5jdGlvbiBpc1Byb21pc2Uob2JqZWN0OiBhbnkpOiBvYmplY3QgaXMgUHJvbWlzZTx1bmtub3duPiB7XG4gIC8vICBXZSBjaG9zZSB0byBub3QgdXNlICdvYmogaW5zdGFuY2VPZiBQcm9taXNlJyBmb3IgdHdvIHJlYXNvbnM6XG4gIC8vICAxLiBJdCBvbmx5IHJlbGlhYmx5IHdvcmtzIGZvciBlczYgUHJvbWlzZSwgbm90IG90aGVyIFByb21pc2VcbiAgLy8gIGltcGxlbWVudGF0aW9ucy5cbiAgLy8gIDIuIEl0IGRvZXNuJ3Qgd29yayB3aXRoIGZyYW1ld29yayB0aGF0IHVzZXMgem9uZS5qcy4gem9uZS5qcyBtb25rZXlcbiAgLy8gIHBhdGNoIHRoZSBhc3luYyBjYWxscywgc28gaXQgaXMgcG9zc2libGUgdGhlIG9iaiAocGF0Y2hlZCkgaXNcbiAgLy8gIGNvbXBhcmluZyB0byBhIHByZS1wYXRjaGVkIFByb21pc2UuXG4gIHJldHVybiBvYmplY3QgJiYgb2JqZWN0LnRoZW4gJiYgdHlwZW9mIG9iamVjdC50aGVuID09PSAnZnVuY3Rpb24nO1xufVxuIl19