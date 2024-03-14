import { dbClient } from '../../constants/dbClient.js';

export const getPlayer = async (baseballReferenceId: string) => {
   return await dbClient.query(
      `
         SELECT
            player.player_id
            ,player.baseball_reference_id
            ,player.bats
            ,player.throws
            ,player.time_born
         FROM
            player
         WHERE
            player.baseball_reference_id = $1
      `,
      [
         baseballReferenceId,
      ],
   )
}