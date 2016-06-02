# Overview

## Pretraining

### Data format

#### states.training.csv

The file format here is:

```
board state,utility
```

where `board state` is of the format `n,n,n,n,n,n,n;n,n,n,n,n,n,n;n,n,n,n,n,n,n;n,n,n,n,n,n,n;n,n,n,n,n,n,n;n,n,n,n,n,n,n` and utility is a real number between -1 and 1. n are either:
* 0 - the space is empty, or
* 1 - player1 is here, or
* 2 - player2 is here.
The semi-colons separate the rows of the board, rows are listed from top to bottom. The full specification for the format is [here](http://theaigames.com/competitions/four-in-a-row/getting-started).

#### states.training.transformed.csv

```
turn,player1board,player2board,utility
```
where `turn` is 0 if it is player1 to play, and 1 if it is player2 to play. `player1board` is 7*6 numbers which are 1 if player1 is here and 0 otherwise. `player2board` is 7*6 numbers which are 1 if player2 is here and 0 otherwise. `utility` is a real number from -1 to 1.

### Working with the data in [Torch](http://torch.ch/)

Install cvsigo:
```
$ luarocks install csvigo
```

See `train.lua`.

I'm not able to get better than a MSE on validation data of 0.05 and I need a MSE of 0.0025. I've had the best results using an MLP of 85->60->30->15->1 layers.