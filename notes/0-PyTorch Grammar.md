# 📕 PyTorch Grammar: The "Cheat Sheet"

Neural network code in Python uses a specialized "short-hand." It is designed to perform math on giant grids of numbers (**Tensors**) with very little code. This guide explains the symbols and terms you will see in our Transformer notes.

---

## 1. The `@` Operator (The Grid Multiplier)
*   **Standard Python (`*`):** Multiplies two single numbers.
*   **PyTorch (`@`):** Performs **Matrix Multiplication**. It multiplies every row of one grid by every column of another grid simultaneously. 
*   **Mental Model:** If `*` is a single step, `@` is a giant leap. It is the core of "Parallel Processing."

## 2. Negative Indexing (`-1`, `-2`)
PyTorch lets you count dimensions from right-to-left using negative numbers.
*   **`0`**: The very first (outermost) dimension.
*   **`-1`**: The very **last** (innermost) dimension.
*   **Why use it?** It is "future-proof." Whether your data is 2D or 100D, `-1` will always find the core data (the actual numbers inside the word's vector).

## 3. Slicing with Colons (`:`)
The colon is used to "slice" through dimensions, like a laser cutter.
*   **`:` (Alone):** "Take everything in this dimension."
*   **`[:T]`**: "Take everything from the start up to index T."
*   **`[2:5]`**: "Take everything from index 2 up to index 4."

**The "Emoji" Code:** `[:, :, :T, :T]` 
> This translates to: *"Take all Batches, take all Heads, but only take the first T words for both the rows and the columns."*

## 4. `.size(-1)` vs. `.shape`
These tell you the "geometry" of your data box.
*   **`.shape`**: Shows the entire dimension list (e.g., `(1, 12, 512)`).
*   **`.size(-1)`**: A command that asks: *"How long is the innermost list of numbers?"* We use this to get the "length" of our word vectors for math scaling.

## 5. `.transpose()` and `.permute()`
These are for "rotating" or "flipping" the data box.
*   **Transpose:** Swaps two dimensions (like flipping a grid so Rows become Columns). We do this in Attention so that Queries and Keys can "face" each other for matchmaking.

## 6. `.view()` and `.reshape()`
This changes the **Shape** of the box without changing the **Data** inside.
*   **Example:** You have a list of 12 numbers. You can `.view(3, 4)` to treat them as a 3x4 grid. It's the same 12 numbers, just organized differently.

## 7. `masked_fill(condition, value)`
The "Search and Replace" tool for Tensors.
*   It looks at a grid, finds every spot that matches your `condition` (like where the "Future" words are), and fills it with a new `value` (like `-infinity`). This is how we "blindfold" the model.

## 8. `nn.Parameter`
This is a "Registration" tag. 
*   In PyTorch, a regular list of numbers is just data. But if you wrap it in `nn.Parameter`, you are telling the computer: **"This is a Weight. Please update and improve this number during training!"**
