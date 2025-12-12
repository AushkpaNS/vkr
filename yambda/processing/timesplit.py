import polars as pl

from ..constants import Constants


def flat_split_train_val_test(
    df: pl.LazyFrame,
    test_timestamp: int,
    val_size: int = 0,
    gap_size: int = Constants.GAP_SIZE,
    drop_non_train_items: bool = False,
    engine: str = "streaming",
    deep: int = None
) -> tuple[pl.LazyFrame, pl.LazyFrame | None, pl.LazyFrame]:
    """
    Splits the dataset into training, validation, and test segments based on the provided timestamps.

    The segments are defined as follows:
    - Training set: [0, test_timestamp - gap_size - val_size - gap_size) if val_size != 0,
                    otherwise [0, test_timestamp - gap_size)
    - Validation set: [test_timestamp - val_size - gap_size, test_timestamp - gap_size), if val_size != 0
    - Test set: [test_timestamp, +inf)

    It retains only those users and items in the validation and test sets that exist in the training set.

    Parameters:
    ----------
    df : LazyFrame
        The dataset in Polars' LazyFrame format.
    test_timestamp : int | None
        The timestamp marking the start of the test set;
    val_size : int | None
        The size of validation. If 0, no validation set is created.
    gap_size : int
        The duration of gap between training and validation/test sets.
    drop_non_train_items : bool
        Whether to drop items that are not in the training set.
    deep : int | None
        If specified, for each user keep only interactions within this time depth 
        from their last interaction in the training period.

    Returns:
    -------
    tuple[LazyFrame, LazyFrame | None, LazyFrame]
        A tuple containing LazyFrames for the training, validation (if applicable), and test sets.
    """

    def drop(df: pl.LazyFrame, unique_train_item_ids) -> pl.LazyFrame:
        if not drop_non_train_items:
            return df

        return (
            df.with_columns(
                pl.col("item_id").is_in(unique_train_item_ids.get_column("item_id").implode()).alias("item_id_in_train")
            )
            .filter("item_id_in_train")
            .drop("item_id_in_train")
        )

    train_timestamp = test_timestamp - gap_size - val_size - (gap_size if val_size != 0 else 0)

    assert gap_size >= 0
    assert val_size >= 0
    assert train_timestamp > 0

    df_lazy = df.lazy()

    """
    # Фильтруем тренировочные данные по временному диапазону v1
    #train = df_lazy.filter(pl.col("timestamp") < train_timestamp)
    if deep is not None:
        train = df_lazy.filter((pl.col("timestamp") < train_timestamp) & (pl.col("timestamp") > train_timestamp - deep))
    else:
        train = df_lazy.filter(pl.col("timestamp") < train_timestamp)
    """

    # Фильтруем тренировочные данные по временному диапазону v2
    train_candidates = df_lazy.filter(pl.col("timestamp") < train_timestamp)
    
    # Если deep задан, фильтруем данные по глубине для каждого пользователя
    if deep is not None and deep > 0:
        # Находим максимальный timestamp для каждого пользователя в тренировочном периоде
        user_max_timestamp = train_candidates.group_by("uid").agg(
            pl.col("timestamp").max().alias("max_timestamp")
        )

        # Присоединяем максимальный timestamp к данным
        train_candidates = train_candidates.join(
            user_max_timestamp.lazy(),
            on="uid",
            how="left"
        )
        
        # Оставляем только взаимодействия, которые попадают в глубину deep
        train = train_candidates.filter(
            pl.when(pl.col("max_timestamp") >= deep)
            .then(pl.col("timestamp") >= pl.col("max_timestamp") - deep)
            .otherwise(True)  # если max_timestamp < deep, включаем все записи пользователя
        ).drop("max_timestamp")
    else:
        train = train_candidates

    unique_train_uids = train.select("uid").unique().collect(engine=engine)
    unique_train_item_ids = train.select("item_id").unique().collect(engine=engine)

    validation = None
    if val_size != 0:
        validation = (
            df_lazy.filter(
                (pl.col("timestamp") >= test_timestamp - val_size - gap_size)
                & (pl.col("timestamp") < test_timestamp - gap_size)
            )
            .with_columns(
                pl.col("uid").is_in(unique_train_uids.get_column("uid").implode()).alias("uid_in_train")
            )  # to prevent filter reordering
            .filter("uid_in_train")
            .drop("uid_in_train")
        )

        validation = drop(validation, unique_train_item_ids)

    test = (
        df_lazy.filter(pl.col("timestamp") >= test_timestamp)
        .with_columns(
            pl.col("uid").is_in(unique_train_uids.get_column("uid").implode()).alias("uid_in_train")
        )  # to prevent filter reordering
        .filter("uid_in_train")
        .drop("uid_in_train")
    )

    test = drop(test, unique_train_item_ids)

    return train, validation, test


def sequential_split_train_val_test(
    df: pl.LazyFrame,
    test_timestamp: int,
    val_size: int = 0,
    gap_size: int = Constants.GAP_SIZE,
    drop_non_train_items: bool = False,
    engine: str = "streaming",
) -> tuple[pl.LazyFrame, pl.LazyFrame | None, pl.LazyFrame]:
    """
    Splits the dataset into training, validation, and test segments based on the provided timestamps.

    The segments are defined as follows:
    - Training set: [0, test_timestamp - gap_size - val_size - gap_size) if val_size != 0,
                    otherwise [0, test_timestamp - gap_size)
    - Validation set: [test_timestamp - val_size - gap_size, test_timestamp - gap_size), if val_size != 0
    - Test set: [test_timestamp, +inf)

    It retains only those users and items in the validation and test sets that exist in the training set.

    Parameters:
    ----------
    df : LazyFrame
        The dataset in Polars' LazyFrame format.
    test_timestamp : int | None
        The timestamp marking the start of the test set;
    val_size : int | None
        The size of validation. If 0, no validation set is created.
    gap_size : int
        The duration of gap between training and validation/test sets.
    drop_non_train_items : bool
        Whether to drop items that are not in the training set.

    Returns:
    -------
    tuple[LazyFrame, LazyFrame | None, LazyFrame]
        A tuple containing LazyFrames for the training, validation (if applicable), and test sets.
    """

    def drop(df: pl.LazyFrame, unique_train_item_ids) -> pl.LazyFrame:
        if not drop_non_train_items:
            return df

        return df.select(
            "uid",
            pl.all()
            .exclude("uid")
            .list.gather(
                pl.col("item_id").list.eval(
                    pl.arg_where(pl.element().is_in(unique_train_item_ids.get_column("item_id").implode()))
                )
            ),
        ).filter(pl.col("item_id").list.len() > 0)

    train_timestamp = test_timestamp - gap_size - val_size - (gap_size if val_size != 0 else 0)

    assert gap_size >= 0
    assert val_size >= 0
    assert train_timestamp > 0

    df_lazy = df.lazy()

    train = df_lazy.select(
        "uid",
        pl.all()
        .exclude("uid")
        .list.gather(pl.col("timestamp").list.eval(pl.arg_where(pl.element() < train_timestamp))),
    ).filter(pl.col("item_id").list.len() > 0)

    unique_train_uids = train.select("uid").unique().collect(engine=engine)
    unique_train_item_ids = train.explode("item_id").select("item_id").unique().collect(engine=engine)

    validation = None
    if val_size != 0:
        validation = (
            df_lazy.select(
                "uid",
                pl.all()
                .exclude("uid")
                .list.gather(
                    pl.col("timestamp").list.eval(
                        pl.arg_where(
                            (pl.element() >= test_timestamp - val_size - gap_size)
                            & (pl.element() < test_timestamp - gap_size)
                        )
                    )
                ),
            )
            .with_columns(
                pl.col("uid").is_in(unique_train_uids.get_column("uid").implode()).alias("uid_in_train")
            )  # to prevent filter reordering
            .filter("uid_in_train")
            .drop("uid_in_train")
        )

        validation = drop(validation, unique_train_item_ids).filter(pl.col("item_id").list.len() > 0)

    test = (
        df_lazy.select(
            "uid",
            pl.all()
            .exclude("uid")
            .list.gather(pl.col("timestamp").list.eval(pl.arg_where(pl.element() >= test_timestamp))),
        )
        #
        .with_columns(
            pl.col("uid").is_in(unique_train_uids.get_column("uid").implode()).alias("uid_in_train")
        )  # to prevent filter reordering
        .filter("uid_in_train")
        .drop("uid_in_train")
    )

    test = drop(test, unique_train_item_ids).filter(pl.col("item_id").list.len() > 0)

    return train, validation, test
