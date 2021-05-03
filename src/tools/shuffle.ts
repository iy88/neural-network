function shuffle(array:any[]) {
	let ks = array.slice(0);
	for (let i = ks[0].length - 1; i > 0; i--) {
		const randomIndex = Math.floor(Math.random() * i + 1);
		for (let j = 0; j < ks.length; j++) {
			[ks[j][i], ks[j][randomIndex]] = [ks[j][randomIndex], ks[j][i]];
		}
	}
	return ks
}

export default shuffle;