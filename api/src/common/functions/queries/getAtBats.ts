import { dbClient } from '../../constants/dbClient.js';

export const getAtBats = async (gameId: number) => {
   return await dbClient.query(
      `
         SELECT
            at_bat.at_bat_id
            ,at_bat.batter_player_id
            ,at_bat.game_id
            ,at_bat.outs
            ,at_bat.pitcher_player_id
            ,at_bat.runs
            ,at_bat.sequence_id
            ,at_bat.total_pitches
         FROM
            at_bat
         WHERE
            at_bat.game_id = $1
      `,
      [
         gameId,
      ],
   )
}