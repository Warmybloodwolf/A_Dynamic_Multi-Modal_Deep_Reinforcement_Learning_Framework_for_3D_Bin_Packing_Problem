First_container_size = """
Given the a list of box data, please choose the most appropriate container size for packing with most space utilization(don't need to pack all the boxes). The box data and container data are in the format of [length, width, height].Notice that the box can be rotated. Several examples are given as follows.

Box data: [[5, 5, 5], [8, 8, 8], [10, 10, 10], [12, 6, 4], [7, 7, 3]]
Container data: [[15, 15, 12], [25, 20, 18]]
>>>>>>
Answer: [15, 15, 12]
------
Box data: [[12, 8, 6], [15, 10, 8], [20, 15, 10], [8, 8, 16], [10, 5, 12]]
Container data: [[25, 20, 15], [35, 30, 25]]
>>>>>>
Answer: [25, 20, 15]
------
Box data: [[3, 3, 3], [4, 4, 4], [6, 6, 6], [5, 3, 8], [4, 2, 6], [3, 3, 9]]
Container data: [[8, 8, 8], [12, 10, 10]]
>>>>>>
Answer: [8, 8, 8]
------
Box data: [[15, 12, 10], [18, 15, 12], [25, 20, 15], [16, 8, 22], [12, 12, 18], [20, 10, 14]]
Container data: [[30, 25, 20], [40, 35, 30]]
>>>>>>
Answer: [30, 25, 20]
------
Box data: [[7, 7, 7], [10, 10, 10], [14, 14, 14], [12, 8, 16], [9, 9, 12], [11, 6, 15], [8, 8, 10]]
Container data: [[18, 16, 16], [25, 22, 20]]
>>>>>>
Answer: [25, 22, 20]
------
Box data: %s
Container data: [[35,23,13], [37,26,13], [38,26,13], [40,28,16], [42, 30, 18], [42,30,40], [52,40,17], [54,45,36]]
>>>>>>
"""
