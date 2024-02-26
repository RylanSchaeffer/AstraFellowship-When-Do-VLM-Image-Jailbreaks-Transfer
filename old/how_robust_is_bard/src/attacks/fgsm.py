from .pgd import PGDAttack


class FGSMAttack(PGDAttack):
    def __init__(self, *args, epsilon=16 / 255, random_start=False, **kwargs):
        kwargs["total_step"] = 1
        kwargs["random_start"] = random_start
        kwargs["epsilon"] = epsilon
        kwargs["step_size"] = epsilon
        super(FGSMAttack, self).__init__(*args, **kwargs)
