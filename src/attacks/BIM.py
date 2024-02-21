from .pgd import PGDAttack


class BIM(PGDAttack):
    def __init__(self, *args, **kwargs):
        kwargs["random_start"] = False
        super(BIM, self).__init__(*args, **kwargs)
