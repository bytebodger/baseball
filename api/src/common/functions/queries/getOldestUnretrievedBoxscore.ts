import { dbClient } from '../../constants/dbClient.js';

export const getOldestUnretrievedBoxscore = async () => {
   return await dbClient.query(`
      SELECT
         web_boxscore.season
         ,web_boxscore.url
         ,web_boxscore.web_boxscore_id
      FROM
         web_boxscore
      WHERE
         web_boxscore.html IS NULL
      AND
         web_boxscore.time_retrieved IS NULL
      ORDER BY
         web_boxscore.web_boxscore_id ASC
      LIMIT 
         1
   `)
}