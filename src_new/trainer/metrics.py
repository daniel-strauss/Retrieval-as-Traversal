import math

# TODO: I admit this was created with AI and it is desaterous. make it logic
# TODO: no type ignore, no shadyness, no setattr 

class _EmaField:
    """Descriptor giving property-like access with automatic EMA tracking.

    ``metrics.pg_loss``    → raw value (float)
    ``metrics.pg_loss_ma`` → exponential moving average (float)
    """

    def __set_name__(self, owner: type, name: str) -> None:
        self._raw = f"_{name}"
        self._ma = f"{name}_ma"

    def __get__(self, obj: object | None, objtype: type | None = None) -> float:
        if obj is None:
            return self  # type: ignore[return-value]
        return getattr(obj, self._raw)

    def __set__(self, obj: object, value: float) -> None:
        setattr(obj, self._raw, value)
        if math.isnan(value):
            return  # NaN does not pollute the MA
        old_ma: float = getattr(obj, self._ma)
        alpha: float = obj._alpha  # type: ignore[attr-defined]
        if math.isnan(old_ma):
            setattr(obj, self._ma, value)
        else:
            setattr(obj, self._ma, alpha * value + (1.0 - alpha) * old_ma)


class Metrics:
    """Per-iteration training metrics with built-in EMA smoothing.

    Created once before the training loop.  Each field assignment updates
    both the raw value and its exponential moving average (``<field>_ma``).

    TODO: consider adding a method to update all fields at once, 
        to avoid multiple EMA updates per iteration.
    """

    pg_loss = _EmaField()
    v_loss = _EmaField()
    entropy_loss = _EmaField()
    internal_pg_loss = _EmaField()
    internal_entropy_loss = _EmaField()
    retrieval_reinforce_loss = _EmaField()
    loss = _EmaField()
    r_loss = _EmaField()
    old_approx_kl = _EmaField()
    approx_kl = _EmaField()
    explained_var = _EmaField()
    clipfrac = _EmaField()
    episode_return = _EmaField()
    episode_length = _EmaField()
    steps_above_min = _EmaField()
    value_mean = _EmaField()
    advantage_mean = _EmaField()

    _EMA_FIELDS = (
        "pg_loss",
        "v_loss",
        "entropy_loss",
        "internal_pg_loss",
        "internal_entropy_loss",
        "retrieval_reinforce_loss",
        "loss",
        "r_loss",
        "old_approx_kl",
        "approx_kl",
        "explained_var",
        "clipfrac",
        "episode_return",
        "episode_length",
        "steps_above_min",
        "value_mean",
        "advantage_mean",
    )

    # MA attributes — set in __init__ via object.__setattr__
    pg_loss_ma: float
    v_loss_ma: float
    entropy_loss_ma: float
    internal_pg_loss_ma: float
    internal_entropy_loss_ma: float
    retrieval_reinforce_loss_ma: float
    loss_ma: float
    r_loss_ma: float
    old_approx_kl_ma: float
    approx_kl_ma: float
    explained_var_ma: float
    clipfrac_ma: float
    episode_return_ma: float
    episode_length_ma: float
    steps_above_min_ma: float
    value_mean_ma: float
    advantage_mean_ma: float

    def __init__(self, alpha: float = 0.1) -> None:
        self._alpha = alpha
        self.additionals: dict[str, dict[str,float]] = {}
        for name in self._EMA_FIELDS:
            object.__setattr__(self, f"{name}_ma", float("nan"))
            object.__setattr__(self, f"_{name}", float("nan"))

    def pass_additional(self, values: dict[str, float]):
        """Update EMA for dynamic metrics and return ``name -> (raw, ma)``.

        Keys already handled by fixed EMA fields are ignored.
        """
        for name, value in values.items():
            if name in self._EMA_FIELDS:
                continue
            
            if name not in self.additionals:
                self.additionals[name] = {'raw':float('nan'), 'ma':float('nan')}    

            self.additionals[name]['raw'] = value

            old_ma = self.additionals[name]['ma'] 
            # TODO weird looking ma handling
            if math.isnan(value):
                ma = old_ma
            elif math.isnan(old_ma):
                ma = value
            else:
                ma = self._alpha * value + (1.0 - self._alpha) * old_ma
            self.additionals[name]['ma'] = ma

            

    def as_dict(self):
        di = {f: getattr(self, f) for f in Metrics._EMA_FIELDS}
        di.update({f"{f}_ma": getattr(self, f"{f}_ma") for f in Metrics._EMA_FIELDS})
        di.update({f"additional/{k}": v['raw'] for k, v in self.additionals.items()})
        di.update({f"additional/{k}_ma": v['ma'] for k, v in self.additionals.items()})
        return di
    

