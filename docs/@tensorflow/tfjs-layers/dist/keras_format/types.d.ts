/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/types" />
/**
 * A value within the JSON-serialized form of a serializable object.
 *
 * The keys of any nested dicts should be in snake_case (i.e., using Python
 * naming conventions) for compatibility with Python Keras.
 *
 * @see PyJsonDict
 */
export type PyJsonValue = boolean | number | string | null | PyJsonArray | PyJsonDict;
/**
 * A key-value dict within the JSON-serialized form of a serializable object.
 *
 * Serialization/deserialization uses stringified-JSON as the storage
 * representation. Typically this should be used for materialized JSON
 * stored on disk or sent/received over the wire.
 *
 * The keys of this dict and of any nested dicts should be in snake_case (i.e.,
 * using Python naming conventions) for compatibility with Python Keras.
 *
 * Internally this is normally converted to a ConfigDict that has CamelCase keys
 * (using TypeScript naming conventions) and support for Enums.
 */
export interface PyJsonDict {
    [key: string]: PyJsonValue;
}
/**
 * A key-value dict like @see PyJsonDict, but with restricted keys.
 *
 * This makes it possible to create subtypes that have only the specified
 * fields, while requiring that the values are JSON-compatible.
 *
 * That is in contrast to extending `PyJsonDict`, or using an intersection type
 * `Foo & PyJsonDict`.  In both of those cases, the fields of Foo are actually
 * allowed to be of types that are incompatible with `PyJsonValue`.  Worse, the
 * index signature of `PyJsonValue` means that *any* key is accepted: eg.
 * `const foo: Foo = ...; foo.bogus = 12; const x = foo.bogus` works for both
 * reading and assignment, even if `bogus` is not a field of the type `Foo`,
 * because the index signature inherited from `PyJsonDict` accepts all strings.
 *
 * Here, we *both* restrict the keys to known values, *and* guarantee that the
 * values associated with those keys are compatible with `PyJsonValue`.
 *
 * This guarantee is easiest to apply via an additional incantation:
 *
 * ```
 * export interface Foo extends PyJson<keyof Foo> {
 *   a: SomeType;
 *   b: SomeOtherType;
 * }
 * ```
 *
 * Now instances of `Foo` have *only* the fields `a` and `b`, and furthermore,
 * if either the type `SomeType` or `SomeOtherType` is incompatible with
 * `PyJsonValue`, the compiler produces a typing error.
 */
export type PyJson<Keys extends string> = {
    [x in Keys]?: PyJsonValue;
};
/**
 * An array of values within the JSON-serialized form of a serializable object.
 *
 * The keys of any nested dicts should be in snake_case (i.e., using Python
 * naming conventions) for compatibility with Python Keras.
 *
 * @see PyJsonDict
 */
export interface PyJsonArray extends Array<PyJsonValue> {
}
/**
 * A Keras JSON entry representing a Keras object such as a Layer.
 *
 * The Keras JSON convention is to provide the `class_name` (e.g., the layer
 * type) at the top level, and then to place the class-specific configuration in
 * a `config` subtree.  These class-specific configurations are provided by
 * subtypes of `PyJsonDict`.  Thus, this `*Serialization` has a type parameter
 * giving the specific type of the wrapped `PyJsonDict`.
 */
export interface BaseSerialization<N extends string, T extends PyJson<Extract<keyof T, string>>> extends PyJsonDict {
    class_name: N;
    config: T;
}
