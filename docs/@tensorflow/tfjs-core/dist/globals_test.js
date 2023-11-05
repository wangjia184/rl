/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import * as tf from './index';
import { ALL_ENVS, describeWithFlags, NODE_ENVS } from './jasmine_util';
import { expectArraysClose } from './test_util';
describe('deprecation warnings', () => {
    beforeEach(() => {
        spyOn(console, 'warn').and.callFake((msg) => null);
    });
    it('deprecationWarn warns', () => {
        // flags_test.ts verifies deprecation warnings are on by default.
        const deprecationVal = tf.env().get('DEPRECATION_WARNINGS_ENABLED');
        tf.env().set('DEPRECATION_WARNINGS_ENABLED', true);
        tf.deprecationWarn('xyz is deprecated.');
        tf.env().set('DEPRECATION_WARNINGS_ENABLED', deprecationVal);
        expect(console.warn).toHaveBeenCalledTimes(1);
        expect(console.warn)
            .toHaveBeenCalledWith('xyz is deprecated. You can disable deprecation warnings with ' +
            'tf.disableDeprecationWarnings().');
    });
    it('disableDeprecationWarnings called, deprecationWarn doesnt warn', () => {
        tf.disableDeprecationWarnings();
        expect(console.warn).toHaveBeenCalledTimes(1);
        expect(console.warn)
            .toHaveBeenCalledWith('TensorFlow.js deprecation warnings have been disabled.');
        // deprecationWarn no longer warns.
        tf.deprecationWarn('xyz is deprecated.');
        expect(console.warn).toHaveBeenCalledTimes(1);
    });
});
describe('Flag flipping methods', () => {
    beforeEach(() => {
        tf.env().reset();
    });
    afterEach(() => {
        tf.env().reset();
    });
    it('tf.enableProdMode', () => {
        tf.enableProdMode();
        expect(tf.env().getBool('PROD')).toBe(true);
    });
    it('tf.enableDebugMode', () => {
        // Silence debug warnings.
        spyOn(console, 'warn');
        tf.enableDebugMode();
        expect(tf.env().getBool('DEBUG')).toBe(true);
    });
});
describeWithFlags('time cpu', NODE_ENVS, () => {
    it('simple upload', async () => {
        const a = tf.zeros([10, 10]);
        const time = await tf.time(() => a.square());
        expect(time.kernelMs).toBeGreaterThan(0);
        expect(time.kernelMs).toBeLessThanOrEqual(time.wallMs);
    });
});
describeWithFlags('tidy', ALL_ENVS, () => {
    it('returns Tensor', async () => {
        tf.tidy(() => {
            const a = tf.tensor1d([1, 2, 3]);
            let b = tf.tensor1d([0, 0, 0]);
            expect(tf.memory().numTensors).toBe(2);
            tf.tidy(() => {
                const result = tf.tidy(() => {
                    b = tf.add(a, b);
                    b = tf.add(a, b);
                    b = tf.add(a, b);
                    return tf.add(a, b);
                });
                // result is new. All intermediates should be disposed.
                expect(tf.memory().numTensors).toBe(2 + 1);
                expect(result.shape).toEqual([3]);
                expect(result.isDisposed).toBe(false);
            });
            // a, b are still here, result should be disposed.
            expect(tf.memory().numTensors).toBe(2);
        });
        expect(tf.memory().numTensors).toBe(0);
    });
    it('multiple disposes does not affect num arrays', () => {
        expect(tf.memory().numTensors).toBe(0);
        const a = tf.tensor1d([1, 2, 3]);
        const b = tf.tensor1d([1, 2, 3]);
        expect(tf.memory().numTensors).toBe(2);
        a.dispose();
        a.dispose();
        expect(tf.memory().numTensors).toBe(1);
        b.dispose();
        expect(tf.memory().numTensors).toBe(0);
    });
    it('allows primitive types', () => {
        const a = tf.tidy(() => 5);
        expect(a).toBe(5);
        const b = tf.tidy(() => 'hello');
        expect(b).toBe('hello');
    });
    it('allows complex types', async () => {
        const res = tf.tidy(() => {
            return { a: tf.scalar(1), b: 'hello', c: [tf.scalar(2), 'world'] };
        });
        expectArraysClose(await res.a.data(), [1]);
        expectArraysClose(await res.c[0].data(), [2]);
    });
    it('returns Tensor[]', async () => {
        const a = tf.tensor1d([1, 2, 3]);
        const b = tf.tensor1d([0, -1, 1]);
        expect(tf.memory().numTensors).toBe(2);
        tf.tidy(() => {
            const result = tf.tidy(() => {
                tf.add(a, b);
                return [tf.add(a, b), tf.sub(a, b)];
            });
            // the 2 results are new. All intermediates should be disposed.
            expect(tf.memory().numTensors).toBe(4);
            expect(result[0].isDisposed).toBe(false);
            expect(result[0].shape).toEqual([3]);
            expect(result[1].isDisposed).toBe(false);
            expect(result[1].shape).toEqual([3]);
            expect(tf.memory().numTensors).toBe(4);
        });
        // the 2 results should be disposed.
        expect(tf.memory().numTensors).toBe(2);
        a.dispose();
        b.dispose();
        expect(tf.memory().numTensors).toBe(0);
    });
    it('basic usage without return', () => {
        const a = tf.tensor1d([1, 2, 3]);
        let b = tf.tensor1d([0, 0, 0]);
        expect(tf.memory().numTensors).toBe(2);
        tf.tidy(() => {
            b = tf.add(a, b);
            b = tf.add(a, b);
            b = tf.add(a, b);
            tf.add(a, b);
        });
        // all intermediates should be disposed.
        expect(tf.memory().numTensors).toBe(2);
    });
    it('nested usage', async () => {
        const a = tf.tensor1d([1, 2, 3]);
        let b = tf.tensor1d([0, 0, 0]);
        expect(tf.memory().numTensors).toBe(2);
        tf.tidy(() => {
            const result = tf.tidy(() => {
                b = tf.add(a, b);
                b = tf.tidy(() => {
                    b = tf.tidy(() => {
                        return tf.add(a, b);
                    });
                    // original a, b, and two intermediates.
                    expect(tf.memory().numTensors).toBe(4);
                    tf.tidy(() => {
                        tf.add(a, b);
                    });
                    // All the intermediates should be cleaned up.
                    expect(tf.memory().numTensors).toBe(4);
                    return tf.add(a, b);
                });
                expect(tf.memory().numTensors).toBe(4);
                return tf.add(a, b);
            });
            expect(tf.memory().numTensors).toBe(3);
            expect(result.isDisposed).toBe(false);
            expect(result.shape).toEqual([3]);
        });
        expect(tf.memory().numTensors).toBe(2);
    });
    it('nested usage returns tensor created from outside scope', () => {
        const x = tf.scalar(1);
        tf.tidy(() => {
            tf.tidy(() => {
                return x;
            });
        });
        expect(x.isDisposed).toBe(false);
    });
    it('nested usage with keep works', () => {
        let b;
        tf.tidy(() => {
            const a = tf.scalar(1);
            tf.tidy(() => {
                b = tf.keep(a);
            });
        });
        expect(b.isDisposed).toBe(false);
        b.dispose();
    });
    it('single argument', () => {
        let hasRan = false;
        tf.tidy(() => {
            hasRan = true;
        });
        expect(hasRan).toBe(true);
    });
    it('single argument, but not a function throws error', () => {
        expect(() => {
            tf.tidy('asdf');
        }).toThrowError();
    });
    it('2 arguments, first is string', () => {
        let hasRan = false;
        tf.tidy('name', () => {
            hasRan = true;
        });
        expect(hasRan).toBe(true);
    });
    it('2 arguments, but first is not string throws error', () => {
        expect(() => {
            // tslint:disable-next-line:no-any
            tf.tidy(4, () => { });
        }).toThrowError();
    });
    it('2 arguments, but second is not a function throws error', () => {
        expect(() => {
            // tslint:disable-next-line:no-any
            tf.tidy('name', 'another name');
        }).toThrowError();
    });
    it('works with arbitrary depth of result', async () => {
        tf.tidy(() => {
            const res = tf.tidy(() => {
                return [tf.scalar(1), [[tf.scalar(2)]], { list: [tf.scalar(3)] }];
            });
            expect(res[0].isDisposed).toBe(false);
            // tslint:disable-next-line:no-any
            expect(res[1][0][0].isDisposed).toBe(false);
            // tslint:disable-next-line:no-any
            expect(res[2].list[0].isDisposed).toBe(false);
            expect(tf.memory().numTensors).toBe(3);
            return res[0];
        });
        // Everything but scalar(1) got disposed.
        expect(tf.memory().numTensors).toBe(1);
    });
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZ2xvYmFsc190ZXN0LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9nbG9iYWxzX3Rlc3QudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxLQUFLLEVBQUUsTUFBTSxTQUFTLENBQUM7QUFDOUIsT0FBTyxFQUFDLFFBQVEsRUFBRSxpQkFBaUIsRUFBRSxTQUFTLEVBQUMsTUFBTSxnQkFBZ0IsQ0FBQztBQUN0RSxPQUFPLEVBQUMsaUJBQWlCLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFFOUMsUUFBUSxDQUFDLHNCQUFzQixFQUFFLEdBQUcsRUFBRTtJQUNwQyxVQUFVLENBQUMsR0FBRyxFQUFFO1FBQ2QsS0FBSyxDQUFDLE9BQU8sRUFBRSxNQUFNLENBQUMsQ0FBQyxHQUFHLENBQUMsUUFBUSxDQUFDLENBQUMsR0FBVyxFQUFRLEVBQUUsQ0FBQyxJQUFJLENBQUMsQ0FBQztJQUNuRSxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyx1QkFBdUIsRUFBRSxHQUFHLEVBQUU7UUFDL0IsaUVBQWlFO1FBQ2pFLE1BQU0sY0FBYyxHQUFHLEVBQUUsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxHQUFHLENBQUMsOEJBQThCLENBQUMsQ0FBQztRQUNwRSxFQUFFLENBQUMsR0FBRyxFQUFFLENBQUMsR0FBRyxDQUFDLDhCQUE4QixFQUFFLElBQUksQ0FBQyxDQUFDO1FBQ25ELEVBQUUsQ0FBQyxlQUFlLENBQUMsb0JBQW9CLENBQUMsQ0FBQztRQUN6QyxFQUFFLENBQUMsR0FBRyxFQUFFLENBQUMsR0FBRyxDQUFDLDhCQUE4QixFQUFFLGNBQWMsQ0FBQyxDQUFDO1FBQzdELE1BQU0sQ0FBQyxPQUFPLENBQUMsSUFBSSxDQUFDLENBQUMscUJBQXFCLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDOUMsTUFBTSxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUM7YUFDZixvQkFBb0IsQ0FDakIsK0RBQStEO1lBQy9ELGtDQUFrQyxDQUFDLENBQUM7SUFDOUMsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsZ0VBQWdFLEVBQUUsR0FBRyxFQUFFO1FBQ3hFLEVBQUUsQ0FBQywwQkFBMEIsRUFBRSxDQUFDO1FBQ2hDLE1BQU0sQ0FBQyxPQUFPLENBQUMsSUFBSSxDQUFDLENBQUMscUJBQXFCLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDOUMsTUFBTSxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUM7YUFDZixvQkFBb0IsQ0FDakIsd0RBQXdELENBQUMsQ0FBQztRQUVsRSxtQ0FBbUM7UUFDbkMsRUFBRSxDQUFDLGVBQWUsQ0FBQyxvQkFBb0IsQ0FBQyxDQUFDO1FBQ3pDLE1BQU0sQ0FBQyxPQUFPLENBQUMsSUFBSSxDQUFDLENBQUMscUJBQXFCLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDaEQsQ0FBQyxDQUFDLENBQUM7QUFDTCxDQUFDLENBQUMsQ0FBQztBQUVILFFBQVEsQ0FBQyx1QkFBdUIsRUFBRSxHQUFHLEVBQUU7SUFDckMsVUFBVSxDQUFDLEdBQUcsRUFBRTtRQUNkLEVBQUUsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxLQUFLLEVBQUUsQ0FBQztJQUNuQixDQUFDLENBQUMsQ0FBQztJQUVILFNBQVMsQ0FBQyxHQUFHLEVBQUU7UUFDYixFQUFFLENBQUMsR0FBRyxFQUFFLENBQUMsS0FBSyxFQUFFLENBQUM7SUFDbkIsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsbUJBQW1CLEVBQUUsR0FBRyxFQUFFO1FBQzNCLEVBQUUsQ0FBQyxjQUFjLEVBQUUsQ0FBQztRQUNwQixNQUFNLENBQUMsRUFBRSxDQUFDLEdBQUcsRUFBRSxDQUFDLE9BQU8sQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQztJQUM5QyxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyxvQkFBb0IsRUFBRSxHQUFHLEVBQUU7UUFDNUIsMEJBQTBCO1FBQzFCLEtBQUssQ0FBQyxPQUFPLEVBQUUsTUFBTSxDQUFDLENBQUM7UUFDdkIsRUFBRSxDQUFDLGVBQWUsRUFBRSxDQUFDO1FBQ3JCLE1BQU0sQ0FBQyxFQUFFLENBQUMsR0FBRyxFQUFFLENBQUMsT0FBTyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO0lBQy9DLENBQUMsQ0FBQyxDQUFDO0FBQ0wsQ0FBQyxDQUFDLENBQUM7QUFFSCxpQkFBaUIsQ0FBQyxVQUFVLEVBQUUsU0FBUyxFQUFFLEdBQUcsRUFBRTtJQUM1QyxFQUFFLENBQUMsZUFBZSxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQzdCLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUM3QixNQUFNLElBQUksR0FBRyxNQUFNLEVBQUUsQ0FBQyxJQUFJLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxDQUFDLE1BQU0sRUFBRSxDQUFDLENBQUM7UUFDN0MsTUFBTSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxlQUFlLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDekMsTUFBTSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxtQkFBbUIsQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLENBQUM7SUFDekQsQ0FBQyxDQUFDLENBQUM7QUFDTCxDQUFDLENBQUMsQ0FBQztBQUVILGlCQUFpQixDQUFDLE1BQU0sRUFBRSxRQUFRLEVBQUUsR0FBRyxFQUFFO0lBQ3ZDLEVBQUUsQ0FBQyxnQkFBZ0IsRUFBRSxLQUFLLElBQUksRUFBRTtRQUM5QixFQUFFLENBQUMsSUFBSSxDQUFDLEdBQUcsRUFBRTtZQUNYLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDakMsSUFBSSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUUvQixNQUFNLENBQUMsRUFBRSxDQUFDLE1BQU0sRUFBRSxDQUFDLFVBQVUsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUN2QyxFQUFFLENBQUMsSUFBSSxDQUFDLEdBQUcsRUFBRTtnQkFDWCxNQUFNLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLEdBQUcsRUFBRTtvQkFDMUIsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO29CQUNqQixDQUFDLEdBQUcsRUFBRSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7b0JBQ2pCLENBQUMsR0FBRyxFQUFFLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztvQkFDakIsT0FBTyxFQUFFLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztnQkFDdEIsQ0FBQyxDQUFDLENBQUM7Z0JBRUgsdURBQXVEO2dCQUN2RCxNQUFNLENBQUMsRUFBRSxDQUFDLE1BQU0sRUFBRSxDQUFDLFVBQVUsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUM7Z0JBQzNDLE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDbEMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUM7WUFDeEMsQ0FBQyxDQUFDLENBQUM7WUFFSCxrREFBa0Q7WUFDbEQsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDekMsQ0FBQyxDQUFDLENBQUM7UUFFSCxNQUFNLENBQUMsRUFBRSxDQUFDLE1BQU0sRUFBRSxDQUFDLFVBQVUsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUN6QyxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyw4Q0FBOEMsRUFBRSxHQUFHLEVBQUU7UUFDdEQsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDdkMsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNqQyxNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ2pDLE1BQU0sQ0FBQyxFQUFFLENBQUMsTUFBTSxFQUFFLENBQUMsVUFBVSxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3ZDLENBQUMsQ0FBQyxPQUFPLEVBQUUsQ0FBQztRQUNaLENBQUMsQ0FBQyxPQUFPLEVBQUUsQ0FBQztRQUNaLE1BQU0sQ0FBQyxFQUFFLENBQUMsTUFBTSxFQUFFLENBQUMsVUFBVSxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3ZDLENBQUMsQ0FBQyxPQUFPLEVBQUUsQ0FBQztRQUNaLE1BQU0sQ0FBQyxFQUFFLENBQUMsTUFBTSxFQUFFLENBQUMsVUFBVSxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3pDLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLHdCQUF3QixFQUFFLEdBQUcsRUFBRTtRQUNoQyxNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQzNCLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFbEIsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxHQUFHLEVBQUUsQ0FBQyxPQUFPLENBQUMsQ0FBQztRQUNqQyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLE9BQU8sQ0FBQyxDQUFDO0lBQzFCLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLHNCQUFzQixFQUFFLEtBQUssSUFBSSxFQUFFO1FBQ3BDLE1BQU0sR0FBRyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsR0FBRyxFQUFFO1lBQ3ZCLE9BQU8sRUFBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsT0FBTyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLEVBQUUsT0FBTyxDQUFDLEVBQUMsQ0FBQztRQUNuRSxDQUFDLENBQUMsQ0FBQztRQUNILGlCQUFpQixDQUFDLE1BQU0sR0FBRyxDQUFDLENBQUMsQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDM0MsaUJBQWlCLENBQUMsTUFBTyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBZSxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUMvRCxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyxrQkFBa0IsRUFBRSxLQUFLLElBQUksRUFBRTtRQUNoQyxNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ2pDLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNsQyxNQUFNLENBQUMsRUFBRSxDQUFDLE1BQU0sRUFBRSxDQUFDLFVBQVUsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUV2QyxFQUFFLENBQUMsSUFBSSxDQUFDLEdBQUcsRUFBRTtZQUNYLE1BQU0sTUFBTSxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsR0FBRyxFQUFFO2dCQUMxQixFQUFFLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztnQkFDYixPQUFPLENBQUMsRUFBRSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUN0QyxDQUFDLENBQUMsQ0FBQztZQUVILCtEQUErRDtZQUMvRCxNQUFNLENBQUMsRUFBRSxDQUFDLE1BQU0sRUFBRSxDQUFDLFVBQVUsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUN2QyxNQUFNLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLFVBQVUsQ0FBQyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQztZQUN6QyxNQUFNLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDckMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUM7WUFDekMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ3JDLE1BQU0sQ0FBQyxFQUFFLENBQUMsTUFBTSxFQUFFLENBQUMsVUFBVSxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3pDLENBQUMsQ0FBQyxDQUFDO1FBRUgsb0NBQW9DO1FBQ3BDLE1BQU0sQ0FBQyxFQUFFLENBQUMsTUFBTSxFQUFFLENBQUMsVUFBVSxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3ZDLENBQUMsQ0FBQyxPQUFPLEVBQUUsQ0FBQztRQUNaLENBQUMsQ0FBQyxPQUFPLEVBQUUsQ0FBQztRQUNaLE1BQU0sQ0FBQyxFQUFFLENBQUMsTUFBTSxFQUFFLENBQUMsVUFBVSxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3pDLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLDRCQUE0QixFQUFFLEdBQUcsRUFBRTtRQUNwQyxNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ2pDLElBQUksQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFL0IsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFdkMsRUFBRSxDQUFDLElBQUksQ0FBQyxHQUFHLEVBQUU7WUFDWCxDQUFDLEdBQUcsRUFBRSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7WUFDakIsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1lBQ2pCLENBQUMsR0FBRyxFQUFFLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztZQUNqQixFQUFFLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUNmLENBQUMsQ0FBQyxDQUFDO1FBRUgsd0NBQXdDO1FBQ3hDLE1BQU0sQ0FBQyxFQUFFLENBQUMsTUFBTSxFQUFFLENBQUMsVUFBVSxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3pDLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLGNBQWMsRUFBRSxLQUFLLElBQUksRUFBRTtRQUM1QixNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ2pDLElBQUksQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFL0IsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFdkMsRUFBRSxDQUFDLElBQUksQ0FBQyxHQUFHLEVBQUU7WUFDWCxNQUFNLE1BQU0sR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLEdBQUcsRUFBRTtnQkFDMUIsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO2dCQUNqQixDQUFDLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxHQUFHLEVBQUU7b0JBQ2YsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsR0FBRyxFQUFFO3dCQUNmLE9BQU8sRUFBRSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7b0JBQ3RCLENBQUMsQ0FBQyxDQUFDO29CQUNILHdDQUF3QztvQkFDeEMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7b0JBRXZDLEVBQUUsQ0FBQyxJQUFJLENBQUMsR0FBRyxFQUFFO3dCQUNYLEVBQUUsQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO29CQUNmLENBQUMsQ0FBQyxDQUFDO29CQUNILDhDQUE4QztvQkFDOUMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7b0JBRXZDLE9BQU8sRUFBRSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7Z0JBQ3RCLENBQUMsQ0FBQyxDQUFDO2dCQUNILE1BQU0sQ0FBQyxFQUFFLENBQUMsTUFBTSxFQUFFLENBQUMsVUFBVSxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUV2QyxPQUFPLEVBQUUsQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1lBQ3RCLENBQUMsQ0FBQyxDQUFDO1lBRUgsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDdkMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUM7WUFDdEMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3BDLENBQUMsQ0FBQyxDQUFDO1FBQ0gsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDekMsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsd0RBQXdELEVBQUUsR0FBRyxFQUFFO1FBQ2hFLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFdkIsRUFBRSxDQUFDLElBQUksQ0FBQyxHQUFHLEVBQUU7WUFDWCxFQUFFLENBQUMsSUFBSSxDQUFDLEdBQUcsRUFBRTtnQkFDWCxPQUFPLENBQUMsQ0FBQztZQUNYLENBQUMsQ0FBQyxDQUFDO1FBQ0wsQ0FBQyxDQUFDLENBQUM7UUFFSCxNQUFNLENBQUMsQ0FBQyxDQUFDLFVBQVUsQ0FBQyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQztJQUNuQyxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyw4QkFBOEIsRUFBRSxHQUFHLEVBQUU7UUFDdEMsSUFBSSxDQUFZLENBQUM7UUFDakIsRUFBRSxDQUFDLElBQUksQ0FBQyxHQUFHLEVBQUU7WUFDWCxNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ3ZCLEVBQUUsQ0FBQyxJQUFJLENBQUMsR0FBRyxFQUFFO2dCQUNYLENBQUMsR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ2pCLENBQUMsQ0FBQyxDQUFDO1FBQ0wsQ0FBQyxDQUFDLENBQUM7UUFFSCxNQUFNLENBQUMsQ0FBQyxDQUFDLFVBQVUsQ0FBQyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQztRQUNqQyxDQUFDLENBQUMsT0FBTyxFQUFFLENBQUM7SUFDZCxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyxpQkFBaUIsRUFBRSxHQUFHLEVBQUU7UUFDekIsSUFBSSxNQUFNLEdBQUcsS0FBSyxDQUFDO1FBQ25CLEVBQUUsQ0FBQyxJQUFJLENBQUMsR0FBRyxFQUFFO1lBQ1gsTUFBTSxHQUFHLElBQUksQ0FBQztRQUNoQixDQUFDLENBQUMsQ0FBQztRQUNILE1BQU0sQ0FBQyxNQUFNLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUM7SUFDNUIsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsa0RBQWtELEVBQUUsR0FBRyxFQUFFO1FBQzFELE1BQU0sQ0FBQyxHQUFHLEVBQUU7WUFDVixFQUFFLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQ2xCLENBQUMsQ0FBQyxDQUFDLFlBQVksRUFBRSxDQUFDO0lBQ3BCLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLDhCQUE4QixFQUFFLEdBQUcsRUFBRTtRQUN0QyxJQUFJLE1BQU0sR0FBRyxLQUFLLENBQUM7UUFDbkIsRUFBRSxDQUFDLElBQUksQ0FBQyxNQUFNLEVBQUUsR0FBRyxFQUFFO1lBQ25CLE1BQU0sR0FBRyxJQUFJLENBQUM7UUFDaEIsQ0FBQyxDQUFDLENBQUM7UUFDSCxNQUFNLENBQUMsTUFBTSxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO0lBQzVCLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLG1EQUFtRCxFQUFFLEdBQUcsRUFBRTtRQUMzRCxNQUFNLENBQUMsR0FBRyxFQUFFO1lBQ1Ysa0NBQWtDO1lBQ2xDLEVBQUUsQ0FBQyxJQUFJLENBQUMsQ0FBUSxFQUFFLEdBQUcsRUFBRSxHQUFFLENBQUMsQ0FBQyxDQUFDO1FBQzlCLENBQUMsQ0FBQyxDQUFDLFlBQVksRUFBRSxDQUFDO0lBQ3BCLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLHdEQUF3RCxFQUFFLEdBQUcsRUFBRTtRQUNoRSxNQUFNLENBQUMsR0FBRyxFQUFFO1lBQ1Ysa0NBQWtDO1lBQ2xDLEVBQUUsQ0FBQyxJQUFJLENBQUMsTUFBTSxFQUFFLGNBQXFCLENBQUMsQ0FBQztRQUN6QyxDQUFDLENBQUMsQ0FBQyxZQUFZLEVBQUUsQ0FBQztJQUNwQixDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyxzQ0FBc0MsRUFBRSxLQUFLLElBQUksRUFBRTtRQUNwRCxFQUFFLENBQUMsSUFBSSxDQUFDLEdBQUcsRUFBRTtZQUNYLE1BQU0sR0FBRyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsR0FBRyxFQUFFO2dCQUN2QixPQUFPLENBQUMsRUFBRSxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBQyxJQUFJLEVBQUUsQ0FBQyxFQUFFLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUMsQ0FBQyxDQUFDO1lBQ2xFLENBQUMsQ0FBQyxDQUFDO1lBQ0gsTUFBTSxDQUFFLEdBQUcsQ0FBQyxDQUFDLENBQWUsQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUM7WUFDckQsa0NBQWtDO1lBQ2xDLE1BQU0sQ0FBRSxHQUFHLENBQUMsQ0FBQyxDQUFTLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsVUFBVSxDQUFDLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDO1lBQ3JELGtDQUFrQztZQUNsQyxNQUFNLENBQUUsR0FBRyxDQUFDLENBQUMsQ0FBUyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUM7WUFDdkQsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDdkMsT0FBTyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDaEIsQ0FBQyxDQUFDLENBQUM7UUFDSCx5Q0FBeUM7UUFDekMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDekMsQ0FBQyxDQUFDLENBQUM7QUFDTCxDQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE5IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0ICogYXMgdGYgZnJvbSAnLi9pbmRleCc7XG5pbXBvcnQge0FMTF9FTlZTLCBkZXNjcmliZVdpdGhGbGFncywgTk9ERV9FTlZTfSBmcm9tICcuL2phc21pbmVfdXRpbCc7XG5pbXBvcnQge2V4cGVjdEFycmF5c0Nsb3NlfSBmcm9tICcuL3Rlc3RfdXRpbCc7XG5cbmRlc2NyaWJlKCdkZXByZWNhdGlvbiB3YXJuaW5ncycsICgpID0+IHtcbiAgYmVmb3JlRWFjaCgoKSA9PiB7XG4gICAgc3B5T24oY29uc29sZSwgJ3dhcm4nKS5hbmQuY2FsbEZha2UoKG1zZzogc3RyaW5nKTogdm9pZCA9PiBudWxsKTtcbiAgfSk7XG5cbiAgaXQoJ2RlcHJlY2F0aW9uV2FybiB3YXJucycsICgpID0+IHtcbiAgICAvLyBmbGFnc190ZXN0LnRzIHZlcmlmaWVzIGRlcHJlY2F0aW9uIHdhcm5pbmdzIGFyZSBvbiBieSBkZWZhdWx0LlxuICAgIGNvbnN0IGRlcHJlY2F0aW9uVmFsID0gdGYuZW52KCkuZ2V0KCdERVBSRUNBVElPTl9XQVJOSU5HU19FTkFCTEVEJyk7XG4gICAgdGYuZW52KCkuc2V0KCdERVBSRUNBVElPTl9XQVJOSU5HU19FTkFCTEVEJywgdHJ1ZSk7XG4gICAgdGYuZGVwcmVjYXRpb25XYXJuKCd4eXogaXMgZGVwcmVjYXRlZC4nKTtcbiAgICB0Zi5lbnYoKS5zZXQoJ0RFUFJFQ0FUSU9OX1dBUk5JTkdTX0VOQUJMRUQnLCBkZXByZWNhdGlvblZhbCk7XG4gICAgZXhwZWN0KGNvbnNvbGUud2FybikudG9IYXZlQmVlbkNhbGxlZFRpbWVzKDEpO1xuICAgIGV4cGVjdChjb25zb2xlLndhcm4pXG4gICAgICAgIC50b0hhdmVCZWVuQ2FsbGVkV2l0aChcbiAgICAgICAgICAgICd4eXogaXMgZGVwcmVjYXRlZC4gWW91IGNhbiBkaXNhYmxlIGRlcHJlY2F0aW9uIHdhcm5pbmdzIHdpdGggJyArXG4gICAgICAgICAgICAndGYuZGlzYWJsZURlcHJlY2F0aW9uV2FybmluZ3MoKS4nKTtcbiAgfSk7XG5cbiAgaXQoJ2Rpc2FibGVEZXByZWNhdGlvbldhcm5pbmdzIGNhbGxlZCwgZGVwcmVjYXRpb25XYXJuIGRvZXNudCB3YXJuJywgKCkgPT4ge1xuICAgIHRmLmRpc2FibGVEZXByZWNhdGlvbldhcm5pbmdzKCk7XG4gICAgZXhwZWN0KGNvbnNvbGUud2FybikudG9IYXZlQmVlbkNhbGxlZFRpbWVzKDEpO1xuICAgIGV4cGVjdChjb25zb2xlLndhcm4pXG4gICAgICAgIC50b0hhdmVCZWVuQ2FsbGVkV2l0aChcbiAgICAgICAgICAgICdUZW5zb3JGbG93LmpzIGRlcHJlY2F0aW9uIHdhcm5pbmdzIGhhdmUgYmVlbiBkaXNhYmxlZC4nKTtcblxuICAgIC8vIGRlcHJlY2F0aW9uV2FybiBubyBsb25nZXIgd2FybnMuXG4gICAgdGYuZGVwcmVjYXRpb25XYXJuKCd4eXogaXMgZGVwcmVjYXRlZC4nKTtcbiAgICBleHBlY3QoY29uc29sZS53YXJuKS50b0hhdmVCZWVuQ2FsbGVkVGltZXMoMSk7XG4gIH0pO1xufSk7XG5cbmRlc2NyaWJlKCdGbGFnIGZsaXBwaW5nIG1ldGhvZHMnLCAoKSA9PiB7XG4gIGJlZm9yZUVhY2goKCkgPT4ge1xuICAgIHRmLmVudigpLnJlc2V0KCk7XG4gIH0pO1xuXG4gIGFmdGVyRWFjaCgoKSA9PiB7XG4gICAgdGYuZW52KCkucmVzZXQoKTtcbiAgfSk7XG5cbiAgaXQoJ3RmLmVuYWJsZVByb2RNb2RlJywgKCkgPT4ge1xuICAgIHRmLmVuYWJsZVByb2RNb2RlKCk7XG4gICAgZXhwZWN0KHRmLmVudigpLmdldEJvb2woJ1BST0QnKSkudG9CZSh0cnVlKTtcbiAgfSk7XG5cbiAgaXQoJ3RmLmVuYWJsZURlYnVnTW9kZScsICgpID0+IHtcbiAgICAvLyBTaWxlbmNlIGRlYnVnIHdhcm5pbmdzLlxuICAgIHNweU9uKGNvbnNvbGUsICd3YXJuJyk7XG4gICAgdGYuZW5hYmxlRGVidWdNb2RlKCk7XG4gICAgZXhwZWN0KHRmLmVudigpLmdldEJvb2woJ0RFQlVHJykpLnRvQmUodHJ1ZSk7XG4gIH0pO1xufSk7XG5cbmRlc2NyaWJlV2l0aEZsYWdzKCd0aW1lIGNwdScsIE5PREVfRU5WUywgKCkgPT4ge1xuICBpdCgnc2ltcGxlIHVwbG9hZCcsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBhID0gdGYuemVyb3MoWzEwLCAxMF0pO1xuICAgIGNvbnN0IHRpbWUgPSBhd2FpdCB0Zi50aW1lKCgpID0+IGEuc3F1YXJlKCkpO1xuICAgIGV4cGVjdCh0aW1lLmtlcm5lbE1zKS50b0JlR3JlYXRlclRoYW4oMCk7XG4gICAgZXhwZWN0KHRpbWUua2VybmVsTXMpLnRvQmVMZXNzVGhhbk9yRXF1YWwodGltZS53YWxsTXMpO1xuICB9KTtcbn0pO1xuXG5kZXNjcmliZVdpdGhGbGFncygndGlkeScsIEFMTF9FTlZTLCAoKSA9PiB7XG4gIGl0KCdyZXR1cm5zIFRlbnNvcicsIGFzeW5jICgpID0+IHtcbiAgICB0Zi50aWR5KCgpID0+IHtcbiAgICAgIGNvbnN0IGEgPSB0Zi50ZW5zb3IxZChbMSwgMiwgM10pO1xuICAgICAgbGV0IGIgPSB0Zi50ZW5zb3IxZChbMCwgMCwgMF0pO1xuXG4gICAgICBleHBlY3QodGYubWVtb3J5KCkubnVtVGVuc29ycykudG9CZSgyKTtcbiAgICAgIHRmLnRpZHkoKCkgPT4ge1xuICAgICAgICBjb25zdCByZXN1bHQgPSB0Zi50aWR5KCgpID0+IHtcbiAgICAgICAgICBiID0gdGYuYWRkKGEsIGIpO1xuICAgICAgICAgIGIgPSB0Zi5hZGQoYSwgYik7XG4gICAgICAgICAgYiA9IHRmLmFkZChhLCBiKTtcbiAgICAgICAgICByZXR1cm4gdGYuYWRkKGEsIGIpO1xuICAgICAgICB9KTtcblxuICAgICAgICAvLyByZXN1bHQgaXMgbmV3LiBBbGwgaW50ZXJtZWRpYXRlcyBzaG91bGQgYmUgZGlzcG9zZWQuXG4gICAgICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1UZW5zb3JzKS50b0JlKDIgKyAxKTtcbiAgICAgICAgZXhwZWN0KHJlc3VsdC5zaGFwZSkudG9FcXVhbChbM10pO1xuICAgICAgICBleHBlY3QocmVzdWx0LmlzRGlzcG9zZWQpLnRvQmUoZmFsc2UpO1xuICAgICAgfSk7XG5cbiAgICAgIC8vIGEsIGIgYXJlIHN0aWxsIGhlcmUsIHJlc3VsdCBzaG91bGQgYmUgZGlzcG9zZWQuXG4gICAgICBleHBlY3QodGYubWVtb3J5KCkubnVtVGVuc29ycykudG9CZSgyKTtcbiAgICB9KTtcblxuICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1UZW5zb3JzKS50b0JlKDApO1xuICB9KTtcblxuICBpdCgnbXVsdGlwbGUgZGlzcG9zZXMgZG9lcyBub3QgYWZmZWN0IG51bSBhcnJheXMnLCAoKSA9PiB7XG4gICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bVRlbnNvcnMpLnRvQmUoMCk7XG4gICAgY29uc3QgYSA9IHRmLnRlbnNvcjFkKFsxLCAyLCAzXSk7XG4gICAgY29uc3QgYiA9IHRmLnRlbnNvcjFkKFsxLCAyLCAzXSk7XG4gICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bVRlbnNvcnMpLnRvQmUoMik7XG4gICAgYS5kaXNwb3NlKCk7XG4gICAgYS5kaXNwb3NlKCk7XG4gICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bVRlbnNvcnMpLnRvQmUoMSk7XG4gICAgYi5kaXNwb3NlKCk7XG4gICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bVRlbnNvcnMpLnRvQmUoMCk7XG4gIH0pO1xuXG4gIGl0KCdhbGxvd3MgcHJpbWl0aXZlIHR5cGVzJywgKCkgPT4ge1xuICAgIGNvbnN0IGEgPSB0Zi50aWR5KCgpID0+IDUpO1xuICAgIGV4cGVjdChhKS50b0JlKDUpO1xuXG4gICAgY29uc3QgYiA9IHRmLnRpZHkoKCkgPT4gJ2hlbGxvJyk7XG4gICAgZXhwZWN0KGIpLnRvQmUoJ2hlbGxvJyk7XG4gIH0pO1xuXG4gIGl0KCdhbGxvd3MgY29tcGxleCB0eXBlcycsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCByZXMgPSB0Zi50aWR5KCgpID0+IHtcbiAgICAgIHJldHVybiB7YTogdGYuc2NhbGFyKDEpLCBiOiAnaGVsbG8nLCBjOiBbdGYuc2NhbGFyKDIpLCAnd29ybGQnXX07XG4gICAgfSk7XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgcmVzLmEuZGF0YSgpLCBbMV0pO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IChyZXMuY1swXSBhcyB0Zi5UZW5zb3IpLmRhdGEoKSwgWzJdKTtcbiAgfSk7XG5cbiAgaXQoJ3JldHVybnMgVGVuc29yW10nLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgYSA9IHRmLnRlbnNvcjFkKFsxLCAyLCAzXSk7XG4gICAgY29uc3QgYiA9IHRmLnRlbnNvcjFkKFswLCAtMSwgMV0pO1xuICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1UZW5zb3JzKS50b0JlKDIpO1xuXG4gICAgdGYudGlkeSgoKSA9PiB7XG4gICAgICBjb25zdCByZXN1bHQgPSB0Zi50aWR5KCgpID0+IHtcbiAgICAgICAgdGYuYWRkKGEsIGIpO1xuICAgICAgICByZXR1cm4gW3RmLmFkZChhLCBiKSwgdGYuc3ViKGEsIGIpXTtcbiAgICAgIH0pO1xuXG4gICAgICAvLyB0aGUgMiByZXN1bHRzIGFyZSBuZXcuIEFsbCBpbnRlcm1lZGlhdGVzIHNob3VsZCBiZSBkaXNwb3NlZC5cbiAgICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1UZW5zb3JzKS50b0JlKDQpO1xuICAgICAgZXhwZWN0KHJlc3VsdFswXS5pc0Rpc3Bvc2VkKS50b0JlKGZhbHNlKTtcbiAgICAgIGV4cGVjdChyZXN1bHRbMF0uc2hhcGUpLnRvRXF1YWwoWzNdKTtcbiAgICAgIGV4cGVjdChyZXN1bHRbMV0uaXNEaXNwb3NlZCkudG9CZShmYWxzZSk7XG4gICAgICBleHBlY3QocmVzdWx0WzFdLnNoYXBlKS50b0VxdWFsKFszXSk7XG4gICAgICBleHBlY3QodGYubWVtb3J5KCkubnVtVGVuc29ycykudG9CZSg0KTtcbiAgICB9KTtcblxuICAgIC8vIHRoZSAyIHJlc3VsdHMgc2hvdWxkIGJlIGRpc3Bvc2VkLlxuICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1UZW5zb3JzKS50b0JlKDIpO1xuICAgIGEuZGlzcG9zZSgpO1xuICAgIGIuZGlzcG9zZSgpO1xuICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1UZW5zb3JzKS50b0JlKDApO1xuICB9KTtcblxuICBpdCgnYmFzaWMgdXNhZ2Ugd2l0aG91dCByZXR1cm4nLCAoKSA9PiB7XG4gICAgY29uc3QgYSA9IHRmLnRlbnNvcjFkKFsxLCAyLCAzXSk7XG4gICAgbGV0IGIgPSB0Zi50ZW5zb3IxZChbMCwgMCwgMF0pO1xuXG4gICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bVRlbnNvcnMpLnRvQmUoMik7XG5cbiAgICB0Zi50aWR5KCgpID0+IHtcbiAgICAgIGIgPSB0Zi5hZGQoYSwgYik7XG4gICAgICBiID0gdGYuYWRkKGEsIGIpO1xuICAgICAgYiA9IHRmLmFkZChhLCBiKTtcbiAgICAgIHRmLmFkZChhLCBiKTtcbiAgICB9KTtcblxuICAgIC8vIGFsbCBpbnRlcm1lZGlhdGVzIHNob3VsZCBiZSBkaXNwb3NlZC5cbiAgICBleHBlY3QodGYubWVtb3J5KCkubnVtVGVuc29ycykudG9CZSgyKTtcbiAgfSk7XG5cbiAgaXQoJ25lc3RlZCB1c2FnZScsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBhID0gdGYudGVuc29yMWQoWzEsIDIsIDNdKTtcbiAgICBsZXQgYiA9IHRmLnRlbnNvcjFkKFswLCAwLCAwXSk7XG5cbiAgICBleHBlY3QodGYubWVtb3J5KCkubnVtVGVuc29ycykudG9CZSgyKTtcblxuICAgIHRmLnRpZHkoKCkgPT4ge1xuICAgICAgY29uc3QgcmVzdWx0ID0gdGYudGlkeSgoKSA9PiB7XG4gICAgICAgIGIgPSB0Zi5hZGQoYSwgYik7XG4gICAgICAgIGIgPSB0Zi50aWR5KCgpID0+IHtcbiAgICAgICAgICBiID0gdGYudGlkeSgoKSA9PiB7XG4gICAgICAgICAgICByZXR1cm4gdGYuYWRkKGEsIGIpO1xuICAgICAgICAgIH0pO1xuICAgICAgICAgIC8vIG9yaWdpbmFsIGEsIGIsIGFuZCB0d28gaW50ZXJtZWRpYXRlcy5cbiAgICAgICAgICBleHBlY3QodGYubWVtb3J5KCkubnVtVGVuc29ycykudG9CZSg0KTtcblxuICAgICAgICAgIHRmLnRpZHkoKCkgPT4ge1xuICAgICAgICAgICAgdGYuYWRkKGEsIGIpO1xuICAgICAgICAgIH0pO1xuICAgICAgICAgIC8vIEFsbCB0aGUgaW50ZXJtZWRpYXRlcyBzaG91bGQgYmUgY2xlYW5lZCB1cC5cbiAgICAgICAgICBleHBlY3QodGYubWVtb3J5KCkubnVtVGVuc29ycykudG9CZSg0KTtcblxuICAgICAgICAgIHJldHVybiB0Zi5hZGQoYSwgYik7XG4gICAgICAgIH0pO1xuICAgICAgICBleHBlY3QodGYubWVtb3J5KCkubnVtVGVuc29ycykudG9CZSg0KTtcblxuICAgICAgICByZXR1cm4gdGYuYWRkKGEsIGIpO1xuICAgICAgfSk7XG5cbiAgICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1UZW5zb3JzKS50b0JlKDMpO1xuICAgICAgZXhwZWN0KHJlc3VsdC5pc0Rpc3Bvc2VkKS50b0JlKGZhbHNlKTtcbiAgICAgIGV4cGVjdChyZXN1bHQuc2hhcGUpLnRvRXF1YWwoWzNdKTtcbiAgICB9KTtcbiAgICBleHBlY3QodGYubWVtb3J5KCkubnVtVGVuc29ycykudG9CZSgyKTtcbiAgfSk7XG5cbiAgaXQoJ25lc3RlZCB1c2FnZSByZXR1cm5zIHRlbnNvciBjcmVhdGVkIGZyb20gb3V0c2lkZSBzY29wZScsICgpID0+IHtcbiAgICBjb25zdCB4ID0gdGYuc2NhbGFyKDEpO1xuXG4gICAgdGYudGlkeSgoKSA9PiB7XG4gICAgICB0Zi50aWR5KCgpID0+IHtcbiAgICAgICAgcmV0dXJuIHg7XG4gICAgICB9KTtcbiAgICB9KTtcblxuICAgIGV4cGVjdCh4LmlzRGlzcG9zZWQpLnRvQmUoZmFsc2UpO1xuICB9KTtcblxuICBpdCgnbmVzdGVkIHVzYWdlIHdpdGgga2VlcCB3b3JrcycsICgpID0+IHtcbiAgICBsZXQgYjogdGYuVGVuc29yO1xuICAgIHRmLnRpZHkoKCkgPT4ge1xuICAgICAgY29uc3QgYSA9IHRmLnNjYWxhcigxKTtcbiAgICAgIHRmLnRpZHkoKCkgPT4ge1xuICAgICAgICBiID0gdGYua2VlcChhKTtcbiAgICAgIH0pO1xuICAgIH0pO1xuXG4gICAgZXhwZWN0KGIuaXNEaXNwb3NlZCkudG9CZShmYWxzZSk7XG4gICAgYi5kaXNwb3NlKCk7XG4gIH0pO1xuXG4gIGl0KCdzaW5nbGUgYXJndW1lbnQnLCAoKSA9PiB7XG4gICAgbGV0IGhhc1JhbiA9IGZhbHNlO1xuICAgIHRmLnRpZHkoKCkgPT4ge1xuICAgICAgaGFzUmFuID0gdHJ1ZTtcbiAgICB9KTtcbiAgICBleHBlY3QoaGFzUmFuKS50b0JlKHRydWUpO1xuICB9KTtcblxuICBpdCgnc2luZ2xlIGFyZ3VtZW50LCBidXQgbm90IGEgZnVuY3Rpb24gdGhyb3dzIGVycm9yJywgKCkgPT4ge1xuICAgIGV4cGVjdCgoKSA9PiB7XG4gICAgICB0Zi50aWR5KCdhc2RmJyk7XG4gICAgfSkudG9UaHJvd0Vycm9yKCk7XG4gIH0pO1xuXG4gIGl0KCcyIGFyZ3VtZW50cywgZmlyc3QgaXMgc3RyaW5nJywgKCkgPT4ge1xuICAgIGxldCBoYXNSYW4gPSBmYWxzZTtcbiAgICB0Zi50aWR5KCduYW1lJywgKCkgPT4ge1xuICAgICAgaGFzUmFuID0gdHJ1ZTtcbiAgICB9KTtcbiAgICBleHBlY3QoaGFzUmFuKS50b0JlKHRydWUpO1xuICB9KTtcblxuICBpdCgnMiBhcmd1bWVudHMsIGJ1dCBmaXJzdCBpcyBub3Qgc3RyaW5nIHRocm93cyBlcnJvcicsICgpID0+IHtcbiAgICBleHBlY3QoKCkgPT4ge1xuICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgdGYudGlkeSg0IGFzIGFueSwgKCkgPT4ge30pO1xuICAgIH0pLnRvVGhyb3dFcnJvcigpO1xuICB9KTtcblxuICBpdCgnMiBhcmd1bWVudHMsIGJ1dCBzZWNvbmQgaXMgbm90IGEgZnVuY3Rpb24gdGhyb3dzIGVycm9yJywgKCkgPT4ge1xuICAgIGV4cGVjdCgoKSA9PiB7XG4gICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgICB0Zi50aWR5KCduYW1lJywgJ2Fub3RoZXIgbmFtZScgYXMgYW55KTtcbiAgICB9KS50b1Rocm93RXJyb3IoKTtcbiAgfSk7XG5cbiAgaXQoJ3dvcmtzIHdpdGggYXJiaXRyYXJ5IGRlcHRoIG9mIHJlc3VsdCcsIGFzeW5jICgpID0+IHtcbiAgICB0Zi50aWR5KCgpID0+IHtcbiAgICAgIGNvbnN0IHJlcyA9IHRmLnRpZHkoKCkgPT4ge1xuICAgICAgICByZXR1cm4gW3RmLnNjYWxhcigxKSwgW1t0Zi5zY2FsYXIoMildXSwge2xpc3Q6IFt0Zi5zY2FsYXIoMyldfV07XG4gICAgICB9KTtcbiAgICAgIGV4cGVjdCgocmVzWzBdIGFzIHRmLlRlbnNvcikuaXNEaXNwb3NlZCkudG9CZShmYWxzZSk7XG4gICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgICBleHBlY3QoKHJlc1sxXSBhcyBhbnkpWzBdWzBdLmlzRGlzcG9zZWQpLnRvQmUoZmFsc2UpO1xuICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgZXhwZWN0KChyZXNbMl0gYXMgYW55KS5saXN0WzBdLmlzRGlzcG9zZWQpLnRvQmUoZmFsc2UpO1xuICAgICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bVRlbnNvcnMpLnRvQmUoMyk7XG4gICAgICByZXR1cm4gcmVzWzBdO1xuICAgIH0pO1xuICAgIC8vIEV2ZXJ5dGhpbmcgYnV0IHNjYWxhcigxKSBnb3QgZGlzcG9zZWQuXG4gICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bVRlbnNvcnMpLnRvQmUoMSk7XG4gIH0pO1xufSk7XG4iXX0=