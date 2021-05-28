export default function autoDiff(fn: Function, x: number) {
  return (fn(x + 1e-10) - fn(x - 1e-10)) / 2e-10;
}