/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import { ENGINE } from '../engine';
import { dispose, tidy } from '../globals';
import { add } from '../ops/add';
import { div } from '../ops/div';
import { mul } from '../ops/mul';
import { sqrt } from '../ops/ops';
import { square } from '../ops/square';
import { zerosLike } from '../ops/zeros_like';
import { Optimizer } from './optimizer';
/** @doclink Optimizer */
export class AdadeltaOptimizer extends Optimizer {
    /** @nocollapse */
    static get className() {
        // Name matters for Python compatibility.
        // This is a getter instead of a property because when it's a property, it
        // prevents the entire class from being tree-shaken.
        return 'Adadelta';
    }
    constructor(learningRate, rho, epsilon = null) {
        super();
        this.learningRate = learningRate;
        this.rho = rho;
        this.epsilon = epsilon;
        this.accumulatedGrads = [];
        this.accumulatedUpdates = [];
        if (epsilon == null) {
            this.epsilon = ENGINE.backend.epsilon();
        }
    }
    applyGradients(variableGradients) {
        const variableNames = Array.isArray(variableGradients) ?
            variableGradients.map(item => item.name) :
            Object.keys(variableGradients);
        variableNames.forEach((name, i) => {
            const value = ENGINE.registeredVariables[name];
            const trainable = false;
            if (this.accumulatedGrads[i] == null) {
                this.accumulatedGrads[i] = {
                    originalName: `${name}/accum_grad`,
                    variable: tidy(() => zerosLike(value).variable(trainable))
                };
            }
            if (this.accumulatedUpdates[i] == null) {
                this.accumulatedUpdates[i] = {
                    originalName: `${name}/accum_var`,
                    variable: tidy(() => zerosLike(value).variable(trainable))
                };
            }
            const gradient = Array.isArray(variableGradients) ?
                variableGradients[i].tensor :
                variableGradients[name];
            if (gradient == null) {
                return;
            }
            const accumulatedGrad = this.accumulatedGrads[i].variable;
            const accumulatedUpdate = this.accumulatedUpdates[i].variable;
            tidy(() => {
                const newAccumulatedGrad = add(mul(accumulatedGrad, this.rho), mul(square(gradient), 1 - this.rho));
                const updates = mul(div(sqrt(add(accumulatedUpdate, this.epsilon)), sqrt(add(accumulatedGrad, this.epsilon))), gradient);
                const newAccumulatedUpdate = add(mul(accumulatedUpdate, this.rho), mul(square(updates), 1 - this.rho));
                accumulatedGrad.assign(newAccumulatedGrad);
                accumulatedUpdate.assign(newAccumulatedUpdate);
                const newValue = add(mul(updates, -this.learningRate), value);
                value.assign(newValue);
            });
        });
        this.incrementIterations();
    }
    dispose() {
        if (this.accumulatedUpdates != null) {
            dispose(this.accumulatedGrads.map(v => v.variable));
            dispose(this.accumulatedUpdates.map(v => v.variable));
        }
    }
    async getWeights() {
        // Order matters for Python compatibility.
        const variables = [...this.accumulatedGrads, ...this.accumulatedUpdates];
        return [await this.saveIterations()].concat(variables.map(v => ({ name: v.originalName, tensor: v.variable })));
    }
    async setWeights(weightValues) {
        weightValues = await this.extractIterations(weightValues);
        const variableCount = weightValues.length / 2;
        const trainable = false;
        this.accumulatedGrads =
            weightValues.slice(0, variableCount).map(v => ({
                originalName: v.name,
                variable: v.tensor.variable(trainable)
            }));
        this.accumulatedUpdates =
            weightValues.slice(variableCount, variableCount * 2)
                .map(v => ({
                originalName: v.name,
                variable: v.tensor.variable(trainable)
            }));
    }
    getConfig() {
        return {
            'learningRate': this.learningRate,
            'rho': this.rho,
            'epsilon': this.epsilon
        };
    }
    /** @nocollapse */
    static fromConfig(cls, config) {
        return new cls(config['learningRate'], config['rho'], config['epsilon']);
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYWRhZGVsdGFfb3B0aW1pemVyLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHRpbWl6ZXJzL2FkYWRlbHRhX29wdGltaXplci50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxPQUFPLEVBQUUsSUFBSSxFQUFDLE1BQU0sWUFBWSxDQUFDO0FBQ3pDLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFDL0IsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLFlBQVksQ0FBQztBQUMvQixPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sWUFBWSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxJQUFJLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFDaEMsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUNyQyxPQUFPLEVBQUMsU0FBUyxFQUFDLE1BQU0sbUJBQW1CLENBQUM7QUFJNUMsT0FBTyxFQUFDLFNBQVMsRUFBb0IsTUFBTSxhQUFhLENBQUM7QUFFekQseUJBQXlCO0FBQ3pCLE1BQU0sT0FBTyxpQkFBa0IsU0FBUSxTQUFTO0lBQzlDLGtCQUFrQjtJQUNsQixNQUFNLEtBQUssU0FBUztRQUNsQix5Q0FBeUM7UUFDekMsMEVBQTBFO1FBQzFFLG9EQUFvRDtRQUNwRCxPQUFPLFVBQVUsQ0FBQztJQUNwQixDQUFDO0lBSUQsWUFDYyxZQUFvQixFQUFZLEdBQVcsRUFDM0MsVUFBa0IsSUFBSTtRQUNsQyxLQUFLLEVBQUUsQ0FBQztRQUZJLGlCQUFZLEdBQVosWUFBWSxDQUFRO1FBQVksUUFBRyxHQUFILEdBQUcsQ0FBUTtRQUMzQyxZQUFPLEdBQVAsT0FBTyxDQUFlO1FBTDVCLHFCQUFnQixHQUF3QixFQUFFLENBQUM7UUFDM0MsdUJBQWtC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QUMsRUFBRSxNQUFNLENBQUMsU0FBUyxDQUFDLENBQUMsQ0FBQztJQUMzRSxDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtkaXNwb3NlLCB0aWR5fSBmcm9tICcuLi9nbG9iYWxzJztcbmltcG9ydCB7YWRkfSBmcm9tICcuLi9vcHMvYWRkJztcbmltcG9ydCB7ZGl2fSBmcm9tICcuLi9vcHMvZGl2JztcbmltcG9ydCB7bXVsfSBmcm9tICcuLi9vcHMvbXVsJztcbmltcG9ydCB7c3FydH0gZnJvbSAnLi4vb3BzL29wcyc7XG5pbXBvcnQge3NxdWFyZX0gZnJvbSAnLi4vb3BzL3NxdWFyZSc7XG5pbXBvcnQge3plcm9zTGlrZX0gZnJvbSAnLi4vb3BzL3plcm9zX2xpa2UnO1xuaW1wb3J0IHtDb25maWdEaWN0LCBTZXJpYWxpemFibGUsIFNlcmlhbGl6YWJsZUNvbnN0cnVjdG9yfSBmcm9tICcuLi9zZXJpYWxpemF0aW9uJztcbmltcG9ydCB7TmFtZWRUZW5zb3IsIE5hbWVkVmFyaWFibGVNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5cbmltcG9ydCB7T3B0aW1pemVyLCBPcHRpbWl6ZXJWYXJpYWJsZX0gZnJvbSAnLi9vcHRpbWl6ZXInO1xuXG4vKiogQGRvY2xpbmsgT3B0aW1pemVyICovXG5leHBvcnQgY2xhc3MgQWRhZGVsdGFPcHRpbWl6ZXIgZXh0ZW5kcyBPcHRpbWl6ZXIge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGdldCBjbGFzc05hbWUoKSB7XG4gICAgLy8gTmFtZSBtYXR0ZXJzIGZvciBQeXRob24gY29tcGF0aWJpbGl0eS5cbiAgICAvLyBUaGlzIGlzIGEgZ2V0dGVyIGluc3RlYWQgb2YgYSBwcm9wZXJ0eSBiZWNhdXNlIHdoZW4gaXQncyBhIHByb3BlcnR5LCBpdFxuICAgIC8vIHByZXZlbnRzIHRoZSBlbnRpcmUgY2xhc3MgZnJvbSBiZWluZyB0cmVlLXNoYWtlbi5cbiAgICByZXR1cm4gJ0FkYWRlbHRhJztcbiAgfVxuICBwcml2YXRlIGFjY3VtdWxhdGVkR3JhZHM6IE9wdGltaXplclZhcmlhYmxlW10gPSBbXTtcbiAgcHJpdmF0ZSBhY2N1bXVsYXRlZFVwZGF0ZXM6IE9wdGltaXplclZhcmlhYmxlW10gPSBbXTtcblxuICBjb25zdHJ1Y3RvcihcbiAgICAgIHByb3RlY3RlZCBsZWFybmluZ1JhdGU6IG51bWJlciwgcHJvdGVjdGVkIHJobzogbnVtYmVyLFxuICAgICAgcHJvdGVjdGVkIGVwc2lsb246IG51bWJlciA9IG51bGwpIHtcbiAgICBzdXBlcigpO1xuXG4gICAgaWYgKGVwc2lsb24gPT0gbnVsbCkge1xuICAgICAgdGhpcy5lcHNpbG9uID0gRU5HSU5FLmJhY2tlbmQuZXBzaWxvbigpO1xuICAgIH1cbiAgfVxuXG4gIGFwcGx5R3JhZGllbnRzKHZhcmlhYmxlR3JhZGllbnRzOiBOYW1lZFZhcmlhYmxlTWFwfE5hbWVkVGVuc29yW10pIHtcbiAgICBjb25zdCB2YXJpYWJsZU5hbWVzID0gQXJyYXkuaXNBcnJheSh2YXJpYWJsZUdyYWRpZW50cykgP1xuICAgICAgICB2YXJpYWJsZUdyYWRpZW50cy5tYXAoaXRlbSA9PiBpdGVtLm5hbWUpIDpcbiAgICAgICAgT2JqZWN0LmtleXModmFyaWFibGVHcmFkaWVudHMpO1xuXG4gICAgdmFyaWFibGVOYW1lcy5mb3JFYWNoKChuYW1lLCBpKSA9PiB7XG4gICAgICBjb25zdCB2YWx1ZSA9IEVOR0lORS5yZWdpc3RlcmVkVmFyaWFibGVzW25hbWVdO1xuICAgICAgY29uc3QgdHJhaW5hYmxlID0gZmFsc2U7XG4gICAgICBpZiAodGhpcy5hY2N1bXVsYXRlZEdyYWRzW2ldID09IG51bGwpIHtcbiAgICAgICAgdGhpcy5hY2N1bXVsYXRlZEdyYWRzW2ldID0ge1xuICAgICAgICAgIG9yaWdpbmFsTmFtZTogYCR7bmFtZX0vYWNjdW1fZ3JhZGAsXG4gICAgICAgICAgdmFyaWFibGU6IHRpZHkoKCkgPT4gemVyb3NMaWtlKHZhbHVlKS52YXJpYWJsZSh0cmFpbmFibGUpKVxuICAgICAgICB9O1xuICAgICAgfVxuICAgICAgaWYgKHRoaXMuYWNjdW11bGF0ZWRVcGRhdGVzW2ldID09IG51bGwpIHtcbiAgICAgICAgdGhpcy5hY2N1bXVsYXRlZFVwZGF0ZXNbaV0gPSB7XG4gICAgICAgICAgb3JpZ2luYWxOYW1lOiBgJHtuYW1lfS9hY2N1bV92YXJgLFxuICAgICAgICAgIHZhcmlhYmxlOiB0aWR5KCgpID0+IHplcm9zTGlrZSh2YWx1ZSkudmFyaWFibGUodHJhaW5hYmxlKSlcbiAgICAgICAgfTtcbiAgICAgIH1cblxuICAgICAgY29uc3QgZ3JhZGllbnQgPSBBcnJheS5pc0FycmF5KHZhcmlhYmxlR3JhZGllbnRzKSA/XG4gICAgICAgICAgdmFyaWFibGVHcmFkaWVudHNbaV0udGVuc29yIDpcbiAgICAgICAgICB2YXJpYWJsZUdyYWRpZW50c1tuYW1lXTtcbiAgICAgIGlmIChncmFkaWVudCA9PSBudWxsKSB7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cblxuICAgICAgY29uc3QgYWNjdW11bGF0ZWRHcmFkID0gdGhpcy5hY2N1bXVsYXRlZEdyYWRzW2ldLnZhcmlhYmxlO1xuICAgICAgY29uc3QgYWNjdW11bGF0ZWRVcGRhdGUgPSB0aGlzLmFjY3VtdWxhdGVkVXBkYXRlc1tpXS52YXJpYWJsZTtcblxuICAgICAgdGlkeSgoKSA9PiB7XG4gICAgICAgIGNvbnN0IG5ld0FjY3VtdWxhdGVkR3JhZCA9XG4gICAgICAgICAgICBhZGQobXVsKGFjY3VtdWxhdGVkR3JhZCwgdGhpcy5yaG8pLFxuICAgICAgICAgICAgICAgIG11bChzcXVhcmUoZ3JhZGllbnQpLCAxIC0gdGhpcy5yaG8pKTtcblxuICAgICAgICBjb25zdCB1cGRhdGVzID1cbiAgICAgICAgICAgIG11bChkaXYoc3FydChhZGQoYWNjdW11bGF0ZWRVcGRhdGUsIHRoaXMuZXBzaWxvbikpLFxuICAgICAgICAgICAgICAgICAgICBzcXJ0KGFkZChhY2N1bXVsYXRlZEdyYWQsIHRoaXMuZXBzaWxvbikpKSxcbiAgICAgICAgICAgICAgICBncmFkaWVudCk7XG5cbiAgICAgICAgY29uc3QgbmV3QWNjdW11bGF0ZWRVcGRhdGUgPVxuICAgICAgICAgICAgYWRkKG11bChhY2N1bXVsYXRlZFVwZGF0ZSwgdGhpcy5yaG8pLFxuICAgICAgICAgICAgICAgIG11bChzcXVhcmUodXBkYXRlcyksIDEgLSB0aGlzLnJobykpO1xuXG4gICAgICAgIGFjY3VtdWxhdGVkR3JhZC5hc3NpZ24obmV3QWNjdW11bGF0ZWRHcmFkKTtcbiAgICAgICAgYWNjdW11bGF0ZWRVcGRhdGUuYXNzaWduKG5ld0FjY3VtdWxhdGVkVXBkYXRlKTtcblxuICAgICAgICBjb25zdCBuZXdWYWx1ZSA9IGFkZChtdWwodXBkYXRlcywgLXRoaXMubGVhcm5pbmdSYXRlKSwgdmFsdWUpO1xuICAgICAgICB2YWx1ZS5hc3NpZ24obmV3VmFsdWUpO1xuICAgICAgfSk7XG4gICAgfSk7XG4gICAgdGhpcy5pbmNyZW1lbnRJdGVyYXRpb25zKCk7XG4gIH1cblxuICBvdmVycmlkZSBkaXNwb3NlKCk6IHZvaWQge1xuICAgIGlmICh0aGlzLmFjY3VtdWxhdGVkVXBkYXRlcyAhPSBudWxsKSB7XG4gICAgICBkaXNwb3NlKHRoaXMuYWNjdW11bGF0ZWRHcmFkcy5tYXAodiA9PiB2LnZhcmlhYmxlKSk7XG4gICAgICBkaXNwb3NlKHRoaXMuYWNjdW11bGF0ZWRVcGRhdGVzLm1hcCh2ID0+IHYudmFyaWFibGUpKTtcbiAgICB9XG4gIH1cblxuICBvdmVycmlkZSBhc3luYyBnZXRXZWlnaHRzKCk6IFByb21pc2U8TmFtZWRUZW5zb3JbXT4ge1xuICAgIC8vIE9yZGVyIG1hdHRlcnMgZm9yIFB5dGhvbiBjb21wYXRpYmlsaXR5LlxuICAgIGNvbnN0IHZhcmlhYmxlczogT3B0aW1pemVyVmFyaWFibGVbXSA9XG4gICAgICAgIFsuLi50aGlzLmFjY3VtdWxhdGVkR3JhZHMsIC4uLnRoaXMuYWNjdW11bGF0ZWRVcGRhdGVzXTtcbiAgICByZXR1cm4gW2F3YWl0IHRoaXMuc2F2ZUl0ZXJhdGlvbnMoKV0uY29uY2F0KFxuICAgICAgICB2YXJpYWJsZXMubWFwKHYgPT4gKHtuYW1lOiB2Lm9yaWdpbmFsTmFtZSwgdGVuc29yOiB2LnZhcmlhYmxlfSkpKTtcbiAgfVxuXG4gIG92ZXJyaWRlIGFzeW5jIHNldFdlaWdodHMod2VpZ2h0VmFsdWVzOiBOYW1lZFRlbnNvcltdKTogUHJvbWlzZTx2b2lkPiB7XG4gICAgd2VpZ2h0VmFsdWVzID0gYXdhaXQgdGhpcy5leHRyYWN0SXRlcmF0aW9ucyh3ZWlnaHRWYWx1ZXMpO1xuICAgIGNvbnN0IHZhcmlhYmxlQ291bnQgPSB3ZWlnaHRWYWx1ZXMubGVuZ3RoIC8gMjtcbiAgICBjb25zdCB0cmFpbmFibGUgPSBmYWxzZTtcbiAgICB0aGlzLmFjY3VtdWxhdGVkR3JhZHMgPVxuICAgICAgICB3ZWlnaHRWYWx1ZXMuc2xpY2UoMCwgdmFyaWFibGVDb3VudCkubWFwKHYgPT4gKHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG9yaWdpbmFsTmFtZTogdi5uYW1lLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdmFyaWFibGU6IHYudGVuc29yLnZhcmlhYmxlKFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRyYWluYWJsZSlcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9KSk7XG4gICAgdGhpcy5hY2N1bXVsYXRlZFVwZGF0ZXMgPVxuICAgICAgICB3ZWlnaHRWYWx1ZXMuc2xpY2UodmFyaWFibGVDb3VudCwgdmFyaWFibGVDb3VudCAqIDIpXG4gICAgICAgICAgICAubWFwKHYgPT4gKHtcbiAgICAgICAgICAgICAgICAgICBvcmlnaW5hbE5hbWU6IHYubmFtZSxcbiAgICAgICAgICAgICAgICAgICB2YXJpYWJsZTogdi50ZW5zb3IudmFyaWFibGUodHJhaW5hYmxlKVxuICAgICAgICAgICAgICAgICB9KSk7XG4gIH1cblxuICBnZXRDb25maWcoKTogQ29uZmlnRGljdCB7XG4gICAgcmV0dXJuIHtcbiAgICAgICdsZWFybmluZ1JhdGUnOiB0aGlzLmxlYXJuaW5nUmF0ZSxcbiAgICAgICdyaG8nOiB0aGlzLnJobyxcbiAgICAgICdlcHNpbG9uJzogdGhpcy5lcHNpbG9uXG4gICAgfTtcbiAgfVxuXG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgb3ZlcnJpZGUgZnJvbUNvbmZpZzxUIGV4dGVuZHMgU2VyaWFsaXphYmxlPihcbiAgICAgIGNsczogU2VyaWFsaXphYmxlQ29uc3RydWN0b3I8VD4sIGNvbmZpZzogQ29uZmlnRGljdCk6IFQge1xuICAgIHJldHVybiBuZXcgY2xzKGNvbmZpZ1snbGVhcm5pbmdSYXRlJ10sIGNvbmZpZ1sncmhvJ10sIGNvbmZpZ1snZXBzaWxvbiddKTtcbiAgfVxufVxuIl19