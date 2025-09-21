from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamban.models.head.ban import UPChannelBAN, DepthwiseBAN, MultiBAN
from siamban.models.head.transhead import TransHead


BANS = {
        'UPChannelBAN': UPChannelBAN,
        'DepthwiseBAN': DepthwiseBAN,
        'MultiBAN': MultiBAN,
        'TransHead': TransHead,
       }


def get_ban_head(name, cfg, **kwargs):
    return BANS[name](**kwargs)

