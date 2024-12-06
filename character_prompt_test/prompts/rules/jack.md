## The game is **Blackjack-11**. Here is the rule:
1. The game is played between a player and a dealer. 
2. The objective of the game is to get as close as possible to a total card value of 11 without exceeding it.
3. Card values: Each hit action will got card value from (1-5).
4. **Outcomes**:
   - The result is determined only when both players choose to "stand," allowing a player to continue hitting even if their hand exceeds 11.  
   - The player closest to 11 without exceeding it wins; otherwise, they lose.  
   - If both exceed 11, it results in a tie, which is preferable to a loss.  
5. **Observed state:**
   In each turn, you will observe the game states:
   - Player_sum: The summation of your hand card. 
   - Dealer_card: The dealer's face-up card value, which can be used to estimate the dealer's final hand.
6. **Player Actions:**
   - 0: "stand": Stop drawing cards and let the dealer play.
   - 1: "hit": Draw one additional card.

### You should:
**Stick to your characteristic**, and decide whether to **hit** or **stand** according to the rules. 
**Stick to your characteristic**, and decide whether to **hit** or **stand** according to the rules. 
**Stick to your characteristic**, and decide whether to **hit** or **stand** according to the rules. 

Give your action following a required form, here are some examples for you:
Example 1: 
I want keep hit because...
Action id: 1.
Example 2:
I want to stand because...
Action id: 0.