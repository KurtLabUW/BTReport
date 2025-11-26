import logging

def get_logger(subject="—"):
    """
    Installs a global logging factory that injects subject ID
    into every emitted log message—even from submodules.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(subject)s] %(message)s",
        force=True  # ensures basicConfig applies even after imports
    )

    old_factory = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.subject = subject  # injected onto every log record
        return record

    logging.setLogRecordFactory(factory)

    return logging.getLogger("btreport")  # optional default logger
