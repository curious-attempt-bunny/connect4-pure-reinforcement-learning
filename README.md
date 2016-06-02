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

```
$ th
th> ds = csvigo.load({path='states.training.transformed.csv', mode='large'})
...
th> #d
42043
th> d[1]
{
  1 : "0"
  2 : "0"
  3 : "0"
  4 : "0"
...
  83 : "0"
  84 : "0"
  85 : "0"
  86 : "-0.004407051282051281"
}
```