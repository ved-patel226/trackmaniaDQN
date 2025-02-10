from pygbx import Gbx, GbxType, CGameChallenge, MapBlock, Vector3

gbx = Gbx(
    "C:/Users/talk2_6h7jpbd/Documents/TrackMania/Tracks/Challenges/My Challenges/AI #3.Challenge.Gbx"
)
challenge = gbx.get_class_by_id(GbxType.CHALLENGE)

for block in challenge.blocks:
    # print(block.position.x, block.position.y, block.position.z)

    print(block)
