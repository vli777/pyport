let input = [1,3,7,20,25,37,59]
let target = 21


const search = (input, target) => {
    let l = 0, r = input.length-1

    if (input[r] <= target) return input[r]
    if (input[l] >= target) return input[l]

    while (l < r) {
        let mid = Math.floor((l+r)/2)

        if (input[mid] > target) {
            r = mid
        } else if (input[mid] < target) {
            l = mid+1
        } else {
            return input[mid]
        }
        console.log(l, r, input[mid])
    }

    return (target - input[l-1] < input[l] - target) ? 
        input[l-1] : input[l]
}

console.log(search(input, target))

const test = [{ hihi: 12312313 }, {yes: 1231232112}, {yes: 123123123123}]
const groupMaps = (maps) => {
    const ans = {}

    maps.forEach(map => {
        for (let [k, v] of Object.entries(map)) {

            if (!ans[k]) {
                let entry = new Set()
                entry.add(v)
                ans[k] = entry
            }
            else ans[k].add(v)
        }
    })

    for (let [k,v] of Object.entries(ans)) {
        ans[k] = Array.from(v)
    }

    return ans
}

console.log(groupMaps(test))