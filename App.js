import { StyleSheet, Text, View } from 'react-native';
import {useState, useEffect} from "react";
import * as ort from "onnxruntime-react-native";


export default function App() {

  const [result, setResult] = useState(null)
  const [session, setSession] = useState(null)

  async function createSession(){
    const newSession = await ort.InferenceSession.create("./xor.onxx");
    setSession(newSession)
  }

  async function callModel(){

    // // prepare inputs. a tensor need its corresponding TypedArray as data
    // const dataA = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    // const dataB = Float32Array.from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]);
    // const tensorA = new ort.Tensor('float32', dataA, [3, 4]);
    // const tensorB = new ort.Tensor('float32', dataB, [4, 3]);

    // // prepare feeds. use model input names as keys.
    const feeds = { a: 1, b: 0 };

    // feed inputs and run
    const results = await session.run(feeds);
    setResult(results)

    // // read from results
    // const dataC = results.c.data;
    // console.log(`data of result tensor 'c': ${dataC}`);
  }

  createSession();
  return (
    <View style={styles.container}>
      <Text>Here the model result will be shown </Text>
      <Text>Result: {results}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
