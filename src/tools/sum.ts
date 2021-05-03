function sum(nums: number[]): number {
  return nums.reduce((total: number, cv: number) => {
    return total + cv
  })
}

export default sum;