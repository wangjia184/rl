/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/// <amd-module name="@tensorflow/tfjs-layers/dist/keras_format/constraint_config" />
import { BaseSerialization } from './types';
export type MaxNormConfig = {
    max_value?: number;
    axis?: number;
};
export type MaxNormSerialization = BaseSerialization<'MaxNorm', MaxNormConfig>;
export type UnitNormConfig = {
    axis?: number;
};
export type UnitNormSerialization = BaseSerialization<'UnitNorm', UnitNormConfig>;
export type NonNegSerialization = BaseSerialization<'NonNeg', null>;
export type MinMaxNormConfig = {
    min_value?: number;
    max_value?: number;
    axis?: number;
    rate?: number;
};
export type MinMaxNormSerialization = BaseSerialization<'MinMaxNorm', MinMaxNormConfig>;
export type ConstraintSerialization = MaxNormSerialization | NonNegSerialization | UnitNormSerialization | MinMaxNormSerialization;
export type ConstraintClassName = ConstraintSerialization['class_name'];
/**
 * A string array of valid Constraint class names.
 *
 * This is guaranteed to match the `ConstraintClassName` union type.
 */
export declare const constraintClassNames: ConstraintClassName[];
