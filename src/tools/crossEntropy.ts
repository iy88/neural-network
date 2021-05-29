function crossEntropy(yp:number,yt:number):number{
  return -yt*Math.log(yp)
}

export default crossEntropy