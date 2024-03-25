import { dbClient } from '../../constants/dbClient.js';

export const getDBTeam = async (abbreviation: string) => {
   return await dbClient.query(
      `
         SELECT
            team.baseball_reference_id
            ,team.team_id
         FROM
            team
         WHERE
            team.baseball_reference_id = $1
      `,
      [
         abbreviation,
      ],
   )
}