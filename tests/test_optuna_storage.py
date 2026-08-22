"""Validation for `src.models._optuna_storage.resolve_storage`.

The regression this guards: `optuna.create_study(storage=<sqlite URI str>)`
builds a pooled SQLAlchemy engine that is never explicitly disposed, which
surfaces as `ResourceWarning: unclosed database` at some later, unrelated
garbage-collection pass -- confusing to debug because the warning's stack
trace points at whatever Optuna call happened to trigger the collection, not
the study that actually leaked the connection.
"""

from __future__ import annotations

import pytest

from src.models._optuna_storage import resolve_storage


class TestResolveStorage:
    def test_sqlite_uri_becomes_an_rdb_storage_with_nullpool(self, tmp_path):
        import optuna
        from sqlalchemy.pool import NullPool

        uri = "sqlite:///" + str(tmp_path / "study.db").replace("\\", "/")
        out = resolve_storage(uri)
        assert isinstance(out, optuna.storages.RDBStorage)
        assert out.engine.pool.__class__ is NullPool

    @pytest.mark.parametrize("value", [
        None,
        "postgresql://localhost/db",
        "mysql://localhost/db",
        "",
    ])
    def test_non_sqlite_values_pass_through_unchanged(self, value):
        assert resolve_storage(value) is value

    def test_an_already_constructed_storage_object_passes_through(self):
        """Not a string at all -- e.g. a caller-built InMemoryStorage."""
        import optuna

        storage = optuna.storages.InMemoryStorage()
        assert resolve_storage(storage) is storage

    def test_resolved_storage_is_actually_usable_by_optuna(self, tmp_path):
        """Not just a NullPool object -- a real study can be created and used."""
        import optuna

        uri = "sqlite:///" + str(tmp_path / "usable.db").replace("\\", "/")
        study = optuna.create_study(storage=resolve_storage(uri))
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=2)
        assert len(study.trials) == 2

    def test_no_resource_warning_on_repeated_short_lived_studies(self, tmp_path, recwarn):
        """The actual regression: many short-lived sqlite studies must not
        leak a connection that later triggers `ResourceWarning: unclosed
        database` when the garbage collector eventually reaps it."""
        import gc

        import optuna

        uri = "sqlite:///" + str(tmp_path / "many.db").replace("\\", "/")
        for i in range(5):
            study = optuna.create_study(
                storage=resolve_storage(uri), study_name=f"s{i}",
            )
            study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=1)
            del study
        gc.collect()
        resource_warnings = [w for w in recwarn.list if issubclass(w.category, ResourceWarning)]
        assert not resource_warnings, [str(w.message) for w in resource_warnings]
