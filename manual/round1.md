We created the correct pdf and cdf of the distribution, we chose two values around the expected value. Our fault was hopign to get accepted rather than calculating the real expected value. 

Solution:
```
https://medium.com/@edgarmaddocks/how-to-trade-with-secretive-fish-036a3411fa74
```
Idea:
Calculate pdf and cdf, make profit function where 1000 - b1 is profit if Y less than b1, and 1000 - b2 if Y between b1 and b2. Take EV of this:

$$EV = (1000 - b_1) (P(Y < b_1)) + (1000 - b_2)(P(b_1 < Y < b_2))$$

now solve for the probabiliteis by integrating the pdf with the relevant bounds

Now you have a 3d function, take partials w.r.t. b1 nad b2, setting equal to zero, and solve for b1 and b2. This gives you decimal values. Run grid search around these values to find optimal profit. 

You can also simulate, by simulating the probability dist'n and running each potential b1 b2 pair (5000 pairs in total bc 100^2 /2)

You will find a similar answer to the closed form solution. Run with 1m or 10m samples to get an accurate answer. 

Answer: b1 = 952, b2 = 978, PnL: 20.4 per fish



