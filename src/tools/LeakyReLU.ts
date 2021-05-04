export default function LeakyReLU(leaky: number): Function {
  return eval(`( ()=> (x) => x > 0 ? x : ${leaky} )()`);
}