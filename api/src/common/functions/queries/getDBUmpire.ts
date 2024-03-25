import { dbClient } from '../../constants/dbClient.js';

export const getDBUmpire = async (name: string) => {
   return await dbClient.query(
      `
         SELECT
            umpire.name
            ,umpire.umpire_id
         FROM
            umpire
         WHERE
            name = $1
      `,
      [
         name,
      ],
   )
}