import ./LAWS.bend as Laws
import Base

# PROOF: both laws by structural induction on `a` (the cached block),
# matching the shape of step_cached()'s own `List.length(F32, cache)`
# split point. Base case (a = Nil{}) additionally splits on `b`'s own
# shape -- List.take/List.drop's n=0n branches still need xs's own
# constructor known to select between "xs is Nil" and "xs is Con",
# even though both give a result equal to `b` either way.

def Laws.take_append_left(a, b):
  match a:
    case Nil{}:
      match b:
        case Nil{}:
          {==}
        case h <> t:
          {==}
    case h <> t:
      %Laws.take_append_left(t, b) : {h<>List.take(&1, F32, List.append(&1, F32, t, b), List.length(&1, F32, t)) == h<>_ : List<F32>}
      {==}

def Laws.drop_append_right(a, b):
  match a:
    case Nil{}:
      match b:
        case Nil{}:
          {==}
        case h <> t:
          {==}
    case h <> t:
      Laws.drop_append_right(t, b)

def main() -> {List.take(&1, F32, List.append(&1, F32, [1.0, 2.0], [3.0, 4.0, 5.0]), 2n) == [1.0, 2.0] : List<F32>}:
  Laws.take_append_left([1.0, 2.0], [3.0, 4.0, 5.0])
