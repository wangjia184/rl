
const url = self.location.toString();
let index = url.lastIndexOf('/worker.js');


const TF_JS_URL = url.substring(0, index) + "/@tensorflow/tfjs/dist/tf.min.js";
const TF_JS_CDN_URL = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.12.0/dist/tf.min.js";
async function load_model() {

    const model = await (async () => {
        try {
            self.postMessage({ type: 'progress', progress: 0.1, message: 'Loading model' });
            return await tf.loadGraphModel('./model/model.json', {
                onProgress: (percent) => {
                    self.postMessage({ type: 'progress', progress: 0.1 + 0.9 * percent, message: 'Loading model' });
                }
            });
        }
        catch (e) {
            self.postMessage({ type: 'error', message: 'Failed to load model.' + e });
            console.log('Failed to load model', e);
            return null;
        }
    })();

    if (!model) {
        self.postMessage({ type: 'error', message: 'Failed to load model' });
        return;
    }


    return model;
}

async function main() {

    try {
        await tf.setBackend('webgl');
        console.log('Successfully loaded WebGL backend');
    } catch {
        await import('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.12.0/dist/tf-backend-wasm.min.js');
        await tf.setBackend('wasm');
        console.log('Successfully loaded WASM backend');
    }



    const model = await load_model();

    self.onmessage = async (evt) => {
        //console.log(evt.data);

        const inputs = tf.tensor(evt.data, [1, 1, 8]/*shape*/, 'float32');
        const tensor = model.predict(inputs);

        // console.log(probabilities[0][0]);
        // [0.994148850440979, 0.005851214751601219]
        // this is a determinate process and hence we don't need sample from distribution
        const probabilities = await tensor.array();
        const flap = probabilities[0][0][1] > probabilities[0][0][0];

        // When using WebGL backend, tf.Tensor memory must be managed explicitly (it is not sufficient to let a tf.Tensor go out of scope for its memory to be released).
        tensor.dispose();
        inputs.dispose();

        self.postMessage({ type: 'prediction', flap : flap });
    };

    self.postMessage({
        type: 'ready',
        parameters: { }
    });






}


self.postMessage({ type: 'progress', progress: 0, message: 'Loading ' + TF_JS_URL });

import(TF_JS_URL)
    .then(async () => {
        await main();
    })
    .catch(async (err) => {

        try {
            await import(TF_JS_CDN_URL);
            await main();
        }
        catch {
            self.postMessage({ type: 'error', message: 'Unable to load ' + TF_JS_URL });
            console.log('Unable to load ' + TF_JS_URL, err);
        }
    });