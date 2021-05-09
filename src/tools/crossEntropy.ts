function crossEntropy(yp:number,yt:number):number{
  return (1/2) * yt * Math.log(yp) + (1 - yt) * Math.log(1 - yp)
}

export default crossEntropy