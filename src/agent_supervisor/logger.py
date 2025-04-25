import logging
import sys

class StdoutLogger:
    def __init__(self, name: str, level: int = logging.INFO):
        """
        Initialise le logger avec un nom et un niveau de log spécifique.

        :param name: Nom du logger.
        :param level: Niveau de log (par défaut: logging.INFO).
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Créer un handler pour stdout
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(level)

        # Créer un formatter et l'ajouter au handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stdout_handler.setFormatter(formatter)

        # Ajouter le handler au logger
        self.logger.addHandler(stdout_handler)

    def debug(self, message: str):
        """
        Envoie un message de niveau DEBUG au logger.

        :param message: Message à logger.
        """
        self.logger.debug(message)

    def info(self, message: str):
        """
        Envoie un message de niveau INFO au logger.

        :param message: Message à logger.
        """
        self.logger.info(message)

    def warning(self, message: str):
        """
        Envoie un message de niveau WARNING au logger.

        :param message: Message à logger.
        """
        self.logger.warning(message)

    def error(self, message: str):
        """
        Envoie un message de niveau ERROR au logger.

        :param message: Message à logger.
        """
        self.logger.error(message)

    def critical(self, message: str):
        """
        Envoie un message de niveau CRITICAL au logger.

        :param message: Message à logger.
        """
        self.logger.critical(message)
